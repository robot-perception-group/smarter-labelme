import cv2
import numpy as np
import copy
import labelme.utils.opencv as ocvutil
from labelme.logger import logger
from labelme.shape import Shape
import torch
import torchvision
from labelme.re3.network import Re3Net
import labelme.re3.bb_util as bb_util
import labelme.re3.im_util as im_util
import labelme.utils.ssdutils as ssdutils
import labelme.utils.ssdmodel as model
import random
from qtpy import QtCore

MAX_TRACK_LENGTH = 16
CROP_SIZE = 227
CROP_PAD = 2
SSDSIZE=300
SSDCROPFACTOR=3.0
SSDTHRESHOLD=0.05
SSDAUTOTHRESHOLD=0.15
SSDMINIOU=0.8
SSDALPHA=0.8
COCOCLASS=['background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def displayImage(image):
    im2=(image*127+128).to(torch.uint8).cpu()
    if len(image.shape)==4:
        im2=im2.squeeze(0)
    t=torchvision.transforms.ToPILImage()
    i=t(im2)
    i.show()

class SSD():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.precision=torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = model.SSD300(backbone=model.ResNet("resnet34"))
        self.model = self.model.to(self.precision)

        def batchnorm_to_float(module):
            """Converts batch norm to FP32"""
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.float()
            for child in module.children():
                batchnorm_to_float(child)
            return module
        if self.device.type == "cpu":
            weights=torch.load("ssd.pt", map_location=lambda storage, loc: storage)["model"]
        else:
            self.model=batchnorm_to_float(self.model)
            weights=torch.load("ssd.pt")["model"]
        self.model.load_state_dict(weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.hardEmpty = torch.empty((0),dtype=self.precision,requires_grad=False).to(self.device)
        self.hardZero = torch.zeros(1,dtype=torch.long,requires_grad=False).to(self.device)
        self.dboxes = ssdutils.dboxes300_coco()
        self.encoder = ssdutils.Encoder(self.dboxes)
        self.resizer = torchvision.transforms.Resize((SSDSIZE,SSDSIZE),interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(self.device)


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
                scores = torch.cat((framelabelscores[indices],self.hardZero.to(self.precision)),0)
                boxes = torch.cat((frameboxes[indices],self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

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

    def prepare(self,image,bbox):
        frame=torch.from_numpy(image).to(self.device).permute(2,0,1).to(self.precision) * (1.0/127.0) - 1.0
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
        return candidates

    def detect(self,candidates,bbox):
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

    def tile(self,image,resolution):
        dim=image.shape[1:]
        cx=int((dim[1]/resolution)*2.5)
        if (cx<2):
            cx=2
        cy=int((dim[0]/resolution)*2.5)
        if (cy<2):
            cy=2
        ix=int((dim[1]-resolution)/(cx-1))
        iy=int((dim[0]-resolution)/(cy-1))
        tiles=[]
        for y in range(cy):
            for x in range(cx):
                tiles.append([x*ix,y*iy,resolution,resolution])
        return tiles

    def alltiles(self,image):
        # top level, inspect whole image
        tiles=[[0,0,image.shape[2],image.shape[1]]]
        # second level, fix aspect ratio
        if image.shape[2]>image.shape[1]:
            tres=int(image.shape[1]*0.95)
        else:
            tres=int(image.shape[2]*0.95)
        # keep reducing 50% - might be skipped for small images
        while tres>(600):
            tiles=tiles+self.tile(image,tres)
            tres=int(tres*0.5)
        # tile at actual network resolution too
        #tiles=tiles+tile(image,networkres)
        return tiles

    def dotile(self,image,tile):
        #we send a crop of the image through the detector network and collect the results...
        subimage=image[:,tile[1]:tile[1]+tile[3],tile[0]:tile[0]+tile[2]]
        subimage2=self.resizer(subimage).unsqueeze(0)
        
        loc_p,conf_p = self.model(subimage2)
        loc_p = loc_p.detach().requires_grad_(False)
        conf_p = conf_p.detach().requires_grad_(False)
        loc_decoded,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)

        conf_p = conf_p[:,:,1:] # ignore label 0 (background)
        return(loc_decoded,conf_p)

    def findnew(self,image,oldrectangles, min_conf=0.01, nms_iou=0.5):
        frame=torch.from_numpy(image).to(self.device).permute(2,0,1).to(self.precision) * (1.0/127.0) - 1.0
        frame=torch.cat([frame[2].unsqueeze(0),frame[1].unsqueeze(0),frame[0].unsqueeze(0)],0)
        w=frame.shape[2]
        h=frame.shape[1]
        exclude=torch.cat((torch.tensor(oldrectangles,dtype=self.precision,device=self.device),self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

        locs=[]
        confs=[]

        aspect=float(w)/float(h)
        if aspect>1.0:
            xfactor=1.0
            yfactor=aspect
        else:
            xfactor=aspect
            yfactor=1.0
        bigframe=torch.zeros((3,SSDSIZE*5,SSDSIZE*5),device=self.device,dtype=self.precision)
        smallframe=self.resizer(frame)  #whole frame to 300x300
        bigframe[:,2*SSDSIZE:3*SSDSIZE,2*SSDSIZE:3*SSDSIZE]=smallframe # insert into here
        for size in [0.9,0.7,0.5]:
            fx=SSDSIZE*xfactor/size
            fy=SSDSIZE*yfactor/size
            x1=int( 2.5*SSDSIZE - 0.5*fx)
            x2=int( 2.5*SSDSIZE + 0.5*fx)
            y1=int( 2.5*SSDSIZE - 0.5*fy)
            y2=int( 2.5*SSDSIZE + 0.5*fy)
            subframe=bigframe[:,y1:y2,x1:x2]
            subframe2=self.resizer(subframe).unsqueeze(0)
            #displayImage(subframe2)
            loc_p,conf_p = self.model(subframe2)
            loc_p = loc_p.detach().requires_grad_(False)
            conf_p = conf_p.detach().requires_grad_(False)
            loc_p,conf_p=self.encoder.scale_back_batch(loc_p,conf_p)
            conf_p = conf_p[:,:,1:] # ignore label 0 (background)

            # correct to actual image (relative size)
            loc_p.view(-1,4)[:,0]*=(fx/SSDSIZE)
            loc_p.view(-1,4)[:,2]*=(fx/SSDSIZE)
            loc_p.view(-1,4)[:,1]*=(fy/SSDSIZE)
            loc_p.view(-1,4)[:,3]*=(fy/SSDSIZE)
            loc_p.view(-1,4)[:,0]+=(x1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,2]+=(x1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,1]+=(y1/SSDSIZE)-2.0
            loc_p.view(-1,4)[:,3]+=(y1/SSDSIZE)-2.0

            # limit to bboxes IN the image
            nloc_p=loc_p.contiguous().squeeze(0)
            nconf_p=conf_p.contiguous().squeeze(0)
            keep=( (nloc_p[:,0]>0.01).logical_and(nloc_p[:,1]>0.01).logical_and(nloc_p[:,2]<0.99).logical_and(nloc_p[:,3]<0.99).logical_and((nloc_p[:,2]-nloc_p[:,0])>0.04).logical_and((nloc_p[:,3]-nloc_p[:,1])>0.04) ).nonzero().flatten()
            loc_p=nloc_p[keep,:].unsqueeze(0)
            conf_p=nconf_p[keep,].unsqueeze(0)

            loc_p.view(-1,4)[:,0]*= w
            loc_p.view(-1,4)[:,2]*= w
            loc_p.view(-1,4)[:,1]*= h
            loc_p.view(-1,4)[:,3]*= h

            locs.append(loc_p)
            confs.append(conf_p)

        bigframe=None
        smallframe=None
        tiles=self.alltiles(frame)
        aresults=[]
        for tile in tiles:
            loc_p,conf_p=self.dotile(frame,tile)
            #aresults.append(torch.tensor([[[18.0,1.0],[tile[0],tile[1]],[tile[0]+tile[2],tile[1]+tile[3]]]]))
            # AVOID BORDER rectangles
            nloc_p=loc_p.contiguous().squeeze(0)
            nconf_p=conf_p.contiguous().squeeze(0)
            keep=( (nloc_p[:,0]>0.01).logical_and(nloc_p[:,1]>0.01).logical_and(nloc_p[:,2]<0.99).logical_and(nloc_p[:,3]<0.99).logical_and((nloc_p[:,2]-nloc_p[:,0])>0.04).logical_and((nloc_p[:,3]-nloc_p[:,1])>0.04) ).nonzero().flatten()
            loc_p=nloc_p[keep,:].unsqueeze(0)
            conf_p=nconf_p[keep,].unsqueeze(0)


            loc_p.view(-1,4)[:,0]*= tile[2]
            loc_p.view(-1,4)[:,2]*= tile[2]
            loc_p.view(-1,4)[:,1]*= tile[3]
            loc_p.view(-1,4)[:,3]*= tile[3]
            loc_p.view(-1,4)[:,0]+= tile[0]
            loc_p.view(-1,4)[:,1]+= tile[1]
            loc_p.view(-1,4)[:,2]+= tile[0]
            loc_p.view(-1,4)[:,3]+= tile[1]
            locs.append(loc_p)
            confs.append(conf_p)
        conf_p=torch.cat(confs,1).contiguous()
        loc_p=torch.cat(locs,1).contiguous()

        #aresults=torch.cat(aresults,dim=0).unsqueeze(0)
        #logger.warn("we now have "+str(aresults.shape))
        #return aresults
        labels=conf_p # background is already removed earlier

        # and now we do nms
        labelmask = labels >= min_conf

        results=[]
        #nms needs to be done the slow way 
        for frame in range(labels.shape[0]):
            lboxes=[self.hardEmpty.view(0,3,2)]
            for label in range(labels.shape[2]):
                framelabelscores=labels[frame,:,label]
                frameboxes=loc_p[frame]
                indices=labelmask[frame,:,label].nonzero().flatten() #this is only needed because of buggy pytorch ONNX export, which doesn't like boolean mask access
                scores = torch.cat((framelabelscores[indices],self.hardZero.to(self.precision)),0)
                boxes = torch.cat((frameboxes[indices],self.hardZero.to(self.precision).view(1,1).expand(1,4)),0)

                # do non maximum suppression
                #index = scores.argsort(descending=True)[0:10000] # only the 10000 highest scores per class are kept, speeds things up in a worst case scenario
                #scores=scores[index].contiguous()
                #boxes=boxes[index].contiguous()
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou) # 2x2 boolean matrix of all boxes with too high jaccard overlap - each row has at least one True value on diagonal
                scorematrix = overlapping.float() * scores[:,None]       #this replaces the boolean values with the scores of the column-box

                keep = (scorematrix.max(dim=0)[0] == scores).nonzero().view(-1)

                # this is serialisable for export to ONNX
                scores = scores[keep].contiguous()
                boxes = boxes[keep].contiguous()

                #PROBLEM: with fp16 some overlapping boxes have the same score - non unique maximum
                #luckily there are not too manx so we just do this again and keep any random one
                overlapping = (ssdutils.calc_iou_tensor(boxes,boxes) > nms_iou)
                rands=torch.randn(scores.view(-1).shape,device=self.device,dtype=self.precision)
                randmatrix = overlapping.float() * rands[:,None]
                keep = (randmatrix.max(dim=0)[0] == rands).nonzero().view(-1)
                
                scores = scores[keep]
                boxes = boxes[keep].view(-1,2,2)

                score_enc = torch.cat( (scores.unsqueeze(1)*0 + label + 1, scores.unsqueeze(1)), 1).unsqueeze(1)
                boxes = torch.cat( (score_enc,boxes), 1)
                # boxes are now of shape [detections][3][2]
                # with content [[class,conf],[x1,y1],[x2,y2]], ...
                lboxes.append(boxes)
            lboxes = torch.cat( lboxes,0 )
            index = lboxes[:,0,1].sort(descending=True)[1]
            finalboxes=lboxes[index].contiguous()

            # make sure these are NOT in exclude
            coords=finalboxes[:,1:,:].view(-1,4)
            overlapping = ssdutils.calc_iou_tensor(exclude,coords) # len(cooords) x len(exclude) matrix
            bad = overlapping>nms_iou
            badexcludes=torch.max(bad,dim=1)[0]
            badcoords=torch.max(bad,dim=0)[0]
            index=badcoords.logical_not().nonzero().flatten()
            results.append(finalboxes[index].contiguous().cpu())

        return results

        

#use a single instance of this, saves memory
SSDMULTIBOX=SSD()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RE3MODEL = Re3Net().to(DEVICE)
RE3PATH="checkpoint.pth"
if DEVICE.type == "cpu":
    RE3MODEL.load_state_dict(torch.load(RE3PATH, map_location=lambda storage, loc: storage))
else:
    RE3MODEL.load_state_dict(torch.load(RE3PATH))

class Re3Tracker(object):
    def __init__(self, model_path = 'checkpoint.pth'):
        self.net=RE3MODEL
        self.device=DEVICE
        self.tracked_data = {}

    def track(self, id, image, prev_image=None, bbox = None, past_bbox=None):

        if bbox is not None:
            # trick - only reinit bounding box, not the whole state
            if id in self.tracked_data:
                lstm_state, initial_state, forward_count = self.tracked_data[id]
            else:
                lstm_state = None
                forward_count = 0
            prev_image = image
            past_bbox = copy.deepcopy(bbox)
        elif id in self.tracked_data:
            lstm_state, initial_state, forward_count = self.tracked_data[id]
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
            
        predicted_bbox = bb_util.from_crop_coordinate_system(network_predicted_bbox.cpu().data.numpy()/10, past_bbox_padded, 1, 1)

        # Reset state
        if forward_count > 0 and forward_count % MAX_TRACK_LENGTH == 0:
            lstm_state = initial_state

        forward_count += 1

        if bbox is not None:
            predicted_bbox = bbox

        predicted_bbox = predicted_bbox.reshape(4)

        self.tracked_data[id] = (lstm_state, initial_state, forward_count)
        
        return predicted_bbox

    def reset(self):
        self.tracked_data = {}

def getRectForTracker(img, shape):
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

class Tracker():

    def __init__(self, *args, **kwargs):
        self.tracker=Re3Tracker()
        self.ref_img = None
        self.shape = None
        self.newshape = None

    @property
    def isRunning(self):
        return (self.ref_img is not None)


    def initTracker(self, qimg,shape):
        status = False
        if qimg.isNull() or not shape:
            logger.warn("invalid tracker initialisation")
            return status
        else:
            fimg = ocvutil.qtImg2CvMat(qimg)
            srect = getRectForTracker(fimg, shape)
            rrect = None
            if (self.newshape is not None):
                rrect = getRectForTracker(fimg, self.newshape)
            logger.info("rectangles:"+str((srect,rrect)))
            if ((rrect is None) or rrect[0]!=srect[0] or rrect[1]!=srect[1] or rrect[2]!=srect[2] or rrect[3]!=srect[3]):
                logger.info("re-initializing tracker")
                _ = self.tracker.track(1,fimg,bbox=np.array([srect[0],srect[1],srect[2],srect[3]]))
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
        srect = getRectForTracker(mimg,self.shape)
        pbox = self.tracker.track(1,mimg,prev_image=self.ref_img,past_bbox=np.array([srect[0],srect[1],srect[2],srect[3]]))
        logger.info("tracker reported:"+str(pbox)+" running SSD")
        prep = SSDMULTIBOX.prepare(mimg,pbox)
        ssd1box = SSDMULTIBOX.detect(prep,pbox)
        ssd2box = SSDMULTIBOX.detect(prep,srect)
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
            trect = getRectForTracker(mimg, result)
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


def trackerAutoAnnotate(qimg,shapes):
    mimg = ocvutil.qtImg2CvMat(qimg)
    rects=[]
    for shape in shapes:
        rects.append(getRectForTracker(mimg,shape))
    newrects=SSDMULTIBOX.findnew(mimg,rects)[-1]
    newrects=newrects[newrects[:,0,1]>SSDAUTOTHRESHOLD]
    id0=int(random.random()*100000)*1000
    id1=0
    newshapes=[]
    for rect in newrects:
        myid=id0+id1
        myname=COCOCLASS[int(rect[0,0])]+("_%d"%(myid))
        shape=Shape(label=myname,shape_type="rectangle",flags={})
        shape.insertPoint(1,QtCore.QPoint(int(rect[1,0]),int(rect[1,1])))
        shape.insertPoint(2,QtCore.QPoint(int(rect[2,0]),int(rect[2,1])))
        id1+=1
        newshapes.append(shape)
    return newshapes
