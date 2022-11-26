import cv2
import numpy as np
import copy
import labelme.utils.opencv as ocvutil
from labelme.logger import logger
import torch
import torchvision
from labelme.re3.network import Re3Net
import labelme.re3.bb_util as bb_util
import labelme.re3.im_util as im_util
import labelme.utils.ssdutils as ssdutils
import labelme.utils.ssdmodel as model


MAX_TRACK_LENGTH = 16
CROP_SIZE = 227
CROP_PAD = 2
SSDSIZE=300
SSDCROPFACTOR=3.0
SSDTHRESHOLD=0.05
SSDMINIOU=0.8
SSDALPHA=0.8

class SSD():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.SSD300(backbone=model.ResNet("resnet34")).half()
        def batchnorm_to_float(module):
            """Converts batch norm to FP32"""
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()
            for child in module.children():
                batchnorm_to_float(child)
            return module
        self.model=batchnorm_to_float(self.model)
        weights=torch.load("horses.pt")["model"]
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.hardEmpty = torch.empty((0),dtype=torch.half,requires_grad=False).to(self.device)
        self.hardZero = torch.zeros(1,dtype=torch.long,requires_grad=False).to(self.device)
        self.dboxes = ssdutils.dboxes300_coco()
        self.encoder = ssdutils.Encoder(self.dboxes)
        self.resizer = torchvision.transforms.Resize((SSDSIZE,SSDSIZE)).to(self.device)


    def inference(self,prediction, min_conf=0.01, nms_iou=0.5):
        loc_p,conf_p = prediction
        loc_p = loc_p.detach().requires_grad_(False)
        conf_p = conf_p.detach().requires_grad_(False) # we don't backpropagate through this

        loc_decoded,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)

        labels = conf_p[:,:,1:] # ignore label 0 (background)

        labelmask = labels >= min_conf

        results=[]
        #nms needs to be done the slow way 
        for frame in range(labels.shape[0]):
            lboxes=[self.hardEmpty.view(0,3,2)]
            for label in range(labels.shape[2]):
                framelabelscores=labels[frame,:,label]
                frameboxes=loc_decoded[frame]
                indices=labelmask[frame,:,label].nonzero().flatten() #this is only needed because of buggy pytorch ONNX export, which doesn't like boolean mask access
                scores = torch.cat((framelabelscores[indices],self.hardZero.half()),0)
                boxes = torch.cat((frameboxes[indices],self.hardZero.half().view(1,1).expand(1,4)),0)

                # do non maximum suppression
                index = scores.argsort(descending=True)[0:200] # only the 200 highest scores per class are kept, speeds things up in a worst case scenario
                scores=scores[index]
                boxes=boxes[index]
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou) # 2x2 boolean matrix of all boxes with too high jaccard overlap - each row has at least one True value on diagonal
                scorematrix = overlapping.float() * scores[:,None]       #this replaces the boolean values with the scores of the column-box

                keep = (scorematrix.max(dim=0)[0] == scores).nonzero().view(-1)

                scores = scores[keep]
                boxes = boxes[keep].view(-1,2,2)
                # this is serialisable for export to ONNX

                score_enc = torch.cat( (scores.unsqueeze(1)*0 + label + 1, scores.unsqueeze(1)), 1).unsqueeze(1)
                boxes = torch.cat( (score_enc,boxes), 1)
                # boxes are now of shape [detections][3][2]
                # with content [[class,conf],[x1,y1],[x2,y2]], ...
                lboxes.append(boxes)
            lboxes = torch.cat( lboxes,0 )
            index = lboxes[:,0,1].sort(descending=True)[1]
            results.append(lboxes[index].contiguous())

        return results

    def detect(self,image,bbox):
        frame=torch.from_numpy(image).to(self.device).permute(2,0,1).half() * (1.0/127.0) - 1.0
        H=frame.shape[1]
        W=frame.shape[2]
        w=bbox[2]-bbox[0]
        h=bbox[3]-bbox[1]
        x0=bbox[0]+w/2
        y0=bbox[1]+h/2
        d=h
        if (w>h):
            d=w
        x1=x0-d*SSDCROPFACTOR*0.5
        x2=x0+d*SSDCROPFACTOR*0.5
        y1=y0-d*SSDCROPFACTOR*0.5
        y2=y0+d*SSDCROPFACTOR*0.5
        x1,y1,x2,y2 = [ (int(a) if a>=0 else 0) for a in [x1,y1,x2,y2]]
        x1,x2 = [ (a if a<=W-1 else W-1) for a in [x1,x2]]
        y1,y2 = [ (a if a<=H-1 else H-1) for a in [y1,y2]]
        subframe=frame[:,y1:y2,x1:x2]
        resized=self.resizer(subframe).unsqueeze(0)
        r0 = self.model(resized)
        r = self.inference(r0)[-1].float()
        r[:,1:,0]*=float(x2-x1)
        r[:,1:,1]*=float(y2-y1)
        r[:,1:,0]+=x1
        r[:,1:,1]+=y1
        candidates=r[:,1:,:].view(-1,4).contiguous()
        candidates=torch.cat((candidates,self.hardZero.float().view(1,1).expand(1,4)),0)
        bboxtensor=torch.tensor(bbox).to(self.device).view(1,4)
        overlapping = (ssdutils.calc_iou_tensor(bboxtensor,candidates)).view(-1)
        keep=torch.argmax(overlapping)
        logger.info(("best ssd box: %f :"%(overlapping[keep]))+str(candidates[keep]))
        if (overlapping[keep]<SSDMINIOU):
            logger.info("BOX is below threshold :( ")
            # ssd found nothing
            return None
        else:
            #ssd DID find something
            logger.info("SSD multibox found box "+str(candidates[keep]))
            return [float(candidates[keep,0]), float(candidates[keep,1]), float(candidates[keep,2]), float(candidates[keep,3])]

#use a single instance of this, saves memory
SSDMULTIBOX=SSD()

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
            logger.info("rectangles:"+str((srect,rrect)))
            if ((rrect is None) or rrect[0]!=srect[0] or rrect[1]!=srect[1] or rrect[2]!=srect[2] or rrect[3]!=srect[3]):
                logger.info("re-initializing tracker")
                _ = self.tracker.track(1,fimg,np.array([srect[0],srect[1],srect[2],srect[3]]))
            else:
                logger.info("keeping tracker")
            self.shape=shape
            self.newshape=None
            self.ref_img=fimg.copy()

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
        logger.info("tracker reported:"+str(pbox)+" running SSD")
        ssd1box = SSDMULTIBOX.detect(mimg,pbox)
        ssd2box = SSDMULTIBOX.detect(mimg,srect)
        if ssd1box is None:
            ssdbox=ssd2box
        elif ssd2box is None:
            ssdbox=ssd1box
        else:
            logger.info("combining ssd boxes")
            ssdbox=list(SSDALPHA*np.array(ssd1box)+(1.0-SSDALPHA)*np.array(ssd2box))
        if ssdbox is not None:
            logger.info("merging with Re3")
            pbox=list(SSDALPHA*np.array(ssdbox)+(1.0-SSDALPHA)*np.array(pbox))
            logger.info("ssd fusion reported:"+str(pbox))

        logger.info("fixed shape:"+str(pbox))
        fw=float(pbox[2]-pbox[0])/float(srect[2]-srect[0])
        fh=float(pbox[3]-pbox[1])/float(srect[3]-srect[1])
        cx=pbox[0]-(srect[0]*fw)
        cy=pbox[1]-(srect[1]*fh)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0,0]=fw
        warp_matrix[1,1]=fh
        warp_matrix[0,2]=cx
        warp_matrix[1,2]=cy
        logger.info("warp matrix:"+str(warp_matrix))

        cc = False

        if (fw==fw and fh==fh):
            result = copy.deepcopy(shape)
            result.transform(warp_matrix)
            trect = self.getRectForTracker(mimg, result)
            logger.info("resulting rect:"+str(trect))
            status = True
            self.newshape=copy.deepcopy(result)
            logger.info("Tracker succeeded")
        else:
            logger.warning("Tracker failed")

        return result , status

    def __reset__(self):
        self.ref_img = None
        self.shape = None
        self.tracker = None

    def stopTracker(self):
        self.__reset__()
