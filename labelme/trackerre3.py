import cv2
import numpy as np
import copy
import labelme.utils.opencv as ocvutil
from labelme.logger import logger
import torch
from labelme.re3.network import Re3Net
import labelme.re3.bb_util as bb_util
import labelme.re3.im_util as im_util


MAX_TRACK_LENGTH = 16
CROP_SIZE = 227
CROP_PAD = 2

class Re3Tracker(object):
    def __init__(self, model_path = 'checkpoint.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Re3Net().to(self.device)
        if model_path is not None:
            if self.device.type == "cpu":
                self.net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            else:
                self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.tracked_data = {}

    def track(self, id, image, bbox = None):
        image = image.copy()

        if bbox is not None:
            # trick - only reinit bounding box, not the whole state
            if id in self.tracked_data:
                lstm_state, initial_state, past_bbox, prev_image, forward_count = self.tracked_data[id]
            else:
                lstm_state = None
                forward_count = 0
            prev_image = image
            past_bbox = copy.deepcopy(bbox)
        elif id in self.tracked_data:
            lstm_state, initial_state, past_bbox, prev_image, forward_count = self.tracked_data[id]
        else:
            raise Exception('Id {0} without initial bounding box'.format(id))

        cropped_input0, past_bbox_padded = im_util.get_cropped_input(prev_image, past_bbox, CROP_PAD, CROP_SIZE)
        cropped_input1, _ = im_util.get_cropped_input(image, past_bbox, CROP_PAD, CROP_SIZE)

        network_input = np.stack((cropped_input0.transpose(2,0,1),cropped_input1.transpose(2,0,1)))
        network_input = torch.tensor(network_input, dtype = torch.float).contiguous()
        
        with torch.no_grad():
            network_input = network_input.to(self.device)
            network_predicted_bbox, lstm_state = self.net(network_input, prevLstmState = lstm_state)

        if forward_count == 0:
            initial_state = lstm_state
            # initial_state = None
            
        prev_image = image

        predicted_bbox = bb_util.from_crop_coordinate_system(network_predicted_bbox.cpu().data.numpy()/10, past_bbox_padded, 1, 1)

        # Reset state
        if forward_count > 0 and forward_count % MAX_TRACK_LENGTH == 0:
            lstm_state = initial_state

        forward_count += 1

        if bbox is not None:
            predicted_bbox = bbox

        predicted_bbox = predicted_bbox.reshape(4)

        self.tracked_data[id] = (lstm_state, initial_state, predicted_bbox, prev_image, forward_count)
        
        return predicted_bbox

    def reset(self):
        self.tracked_data = {}

class Tracker():

    def __init__(self, *args, **kwargs):
        self.tracker=Re3Tracker()
        self.ref_img = None
        self.shape = None
        self.newshape = None

    @property
    def isRunning(self):
        return (self.ref_img is not None)

    def getRectForTracker(self, img, shape):
        H = img.shape[0]
        W = img.shape[1]

        qrect = shape.boundingRect()
        tl = qrect.topLeft()
        h = qrect.height()
        w = qrect.width()

        x = tl.x()
        y = tl.y()
        x2 = x+w
        y2 = y+h
        x,y,x2,y2 = [ (a if a>=0 else 0) for a in [x,y,x2,y2]]
        x,x2 = [ (a if a<=W-1 else W-1) for a in [x,x2]]
        y,y2 = [ (a if a<=H-1 else H-1) for a in [y,y2]]

        x= x if x>=0 else 0
        y= y if y>=0 else 0

        return [int(_) for _ in [x,y,x2,y2]]

    def initTracker(self, qimg,shape):
        status = False
        if qimg.isNull() or not shape:
            logger.warn("invalid tracker initialisation")
            return status
        else:
            fimg = ocvutil.qtImg2CvMat(qimg)
            srect = self.getRectForTracker(fimg, shape)
            rrect = None
            if (self.newshape is not None):
                rrect = self.getRectForTracker(fimg, self.newshape)
            logger.warn("rectangles:"+str((srect,rrect)))
            if ((rrect is None) or rrect[0]!=srect[0] or rrect[1]!=srect[1] or rrect[2]!=srect[2] or rrect[3]!=srect[3]):
                logger.warn("re-initializing tracker")
                _ = self.tracker.track(1,fimg,np.array([srect[0],srect[1],srect[2],srect[3]]))
            else:
                logger.warn("keeping tracker")
            self.shape=shape
            self.newshape=None
            self.ref_img=True

            status = True

        return status

    def updateTracker(self, qimg, shape):

        assert (shape and shape.label == self.shape.label),"Invalid tracker state!"

        status = False

        result = shape

        if not self.isRunning or qimg.isNull():
            if not self.isRunning:
                logger.warning("tracker is not running")
            else:
                logger.warning("no image")
            return result, status

        mimg = ocvutil.qtImg2CvMat(qimg)
        srect = self.getRectForTracker(mimg,self.shape)
        pbox = self.tracker.track(1,mimg)
        logger.warn("tracker reported:"+str(pbox))
        # stupid re 3 reports bbox the wrong way around... why???
        #if pbox[2]<pbox[0]:
        #    x=pbox[2]
        #    pbox[2]=pbox[0]
        #    pbox[0]=x
        #if pbox[3]<pbox[1]:
        #    y=pbox[3]
        #    pbox[3]=pbox[1]
        #    pbox[1]=y
        w=pbox[2]-pbox[0] # transform to x,y,w,h
        h=pbox[3]-pbox[1]
        logger.warn("fixed shape:"+str(pbox))
        fw=float(pbox[2]-pbox[0])/w
        fh=float(pbox[3]-pbox[1])/h
        cx=pbox[0]-(srect[0]*fw)
        cy=pbox[1]-(srect[1]*fh)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0,0]=fw
        warp_matrix[1,1]=fh
        warp_matrix[0,2]=cx
        warp_matrix[1,2]=cy
        logger.warn("warp matrix:"+str(warp_matrix))

        cc = False

        if (fw==fw and fh==fh):
            result = copy.deepcopy(shape)
            result.transform(warp_matrix)
            trect = self.getRectForTracker(mimg, result)
            print("resulting rect:"+str(trect))
            status = True
            self.newshape=copy.deepcopy(result)
            logger.warning("Tracker succeeded")
        else:
            logger.warning("Tracker failed")

        return result , status

    def __reset__(self):
        self.ref_img = None
        self.shape = None
        self.tracker = None

    def stopTracker(self):
        self.__reset__()
