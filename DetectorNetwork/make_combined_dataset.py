#!/usr/bin/env python3

# read in multiple datasets from smarter labelme
# generate annotated crops for
# training
# testing
# training set is combined with MSCOCO
# annotations stored in MSCOCO format

import argparse
import os
import re
import json
import numpy as np
import torch
import torchvision
import copy
import shutil

def add(array,key,num):
    if key in array:
        array[key]+=num
    else:
        array[key]=num

def loadImage(filename):
    return torchvision.io.read_image(filename)

def saveImage(filename,image):
    return torchvision.io.write_jpeg(image,filename,99)

def scale(image,dimensions):
    w,h=dimensions
    t=torchvision.transforms.Resize((h,w),torchvision.transforms.InterpolationMode.BILINEAR,antialias=True)
    return t(image)

def crop(image,dimensions):
    x,y,w,h=dimensions
    l=x
    r=l+w
    t=y
    b=t+h
    return image[:,t:b,l:r]

def main():
    parser = argparse.ArgumentParser(description='Create Combined Dataset for Detection from SmarterLabelme dataset and MSCOCO')
    parser.add_argument('folder', type=str, nargs='+', help='folder to analyse')
    parser.add_argument('destination_folder', type=str, help='folder to store images and json')
    parser.add_argument('--width', type=int, help='dimensions for image', default=640)
    parser.add_argument('--height', type=int, help='dimensions for image', default=480)
    parser.add_argument('--min-size', type=int, help='minimum size of a crop in pixels', default=150)
    parser.add_argument('--datasize', type=int, help='how much data to create', default=20000)
    parser.add_argument('--random', type=int, help='random seed for dataset generation', default=1337)
    parser.add_argument('--classes', type=str, help='dictionary of optional class mappings', default="{}")
    parser.add_argument('--test-percentage', type=float, help='Percentage of data reserved for test set', default=5)
    parser.add_argument('--min-anno-size', type=float, help='Minimum size for an annotation to be considered', default=3)
    parser.add_argument('--cocofolder', type=str, help='folder with coco dataset to combine with (only train)', default="")

    args = parser.parse_args()

    rng=np.random.default_rng(seed=args.random)

    filelists={}
    for o in args.folder:
        if not os.path.isdir(o+'/Annotations'):
            print("Error: source annotation folder %s cannot be opened\n"%o)
            exit(1)
        try:
            filelists[o]=sorted([f for f in os.listdir(o+'/Annotations') if (re.search("\.json$",f) is not None)])
        except:
            print("Error: Could not get annotations from %s\n"%o)
            exit(1)

    if not os.path.isdir(args.destination_folder):
        print("Error: destinationfolder %s cannot be opened\n"%args.destination_folder)
        exit(1)
    try:
        args_classes=json.loads(args.classes)
    except Exception as e:
        print("Error parsing classes - invalid json string: %s - %s"%(args.classes,e))
        exit(1)
    if not isinstance(args_classes,dict):
        print("Error parsing classes - %s is not a dictionary"%args.classes)
        exit(1)

    ids={}
    template=json.load(open(os.path.abspath(os.path.dirname(__file__))+'/template.json'))
    annotations={'train2017':{},'val2017':{}}
    classes={}
    dclasses={}
    for c in template['categories']:
        dclasses[c['name']]=c

    for o in filelists:
        for f in filelists[o]:
            frame_id=int(re.search("([0-9]*)\.json",f)[1])
            jdata=json.load(open(o+'/Annotations/'+f))
            dset='train2017'
            if rng.random()*100<args.test_percentage:
                dset='val2017'
            annotations[dset][frame_id]=jdata
            jdata['frame_id']=frame_id
            jdata['frame_file']=f
            jdata['frame_folder']=o
            jdata['useful_shapes']=[]
            for shape in jdata['shapes']:
                add(ids,shape["label"],1)
                #substitute classname
                rc=None
                c=shape["label"]
                for rcc in args_classes:
                    if c.startswith(rcc):
                        c=args_classes[rcc]
                #find class
                c=re.subn('_',' ',c)[0]
                for tc in dclasses:
                    if c.startswith(tc+' '):
                        rc=dclasses[tc]
                if rc is not None:
                    shape['class']=rc
                    jdata['useful_shapes'].append(shape)
                    add(classes,rc['name'],1)

    print("Classes:")
    print(json.dumps(classes,indent=1))
    print("Instances:")
    print(json.dumps(ids,indent=1))

    da={'train2017':copy.deepcopy(template),'val2017':copy.deepcopy(template)}
    try:
        for dset in da:
            os.mkdir(args.destination_folder+'/'+dset)
        os.mkdir(args.destination_folder+'/annotations')
    except:
        pass

    for dset in da:
        print("extracting framecrops for %s"%dset)
        annoid=0
        for cid in range(args.datasize if dset=='train2017' else int(args.datasize*(args.test_percentage/100))):
            print(cid)
            found_annotations=0
            while found_annotations==0:
                frameid=rng.choice(list(annotations[dset].keys()))
                frame=annotations[dset][frameid]
                width=rng.integers(frame['imageWidth']-args.min_size)+args.min_size
                height=int((width*args.height)/args.width)
                if height>frame['imageHeight']:
                    height=frame['imageHeight']
                    width=int((height*args.width)/args.height)
                offsetx=rng.integers(1+frame['imageWidth']-width)
                offsety=rng.integers(1+frame['imageHeight']-height)
                scalefactor=args.width/float(width)

                for anno in frame['useful_shapes']:

                    bbox=[min(anno['points'][0][0],anno['points'][1][0]),
                          min(anno['points'][0][1],anno['points'][1][1]),
                          abs(anno['points'][0][0]-anno['points'][1][0]),
                          abs(anno['points'][0][1]-anno['points'][1][1])]
                    if bbox[0]+bbox[2]>=offsetx and \
                        bbox[1]+bbox[3]>=offsety and \
                        bbox[0]<offsetx+width and \
                        bbox[1]<offsety+height:
                            # count only these annotations that are for the correct image and within the crop
                            bbox[0]-=offsetx
                            bbox[1]-=offsety
                            if bbox[0]<0:
                                bbox[2]+=bbox[0]
                                bbox[0]=0
                            if bbox[1]<0:
                                bbox[3]+=bbox[1]
                                bbox[1]=0
                            if bbox[0]+bbox[2]>width:
                                bbox[2]=width-bbox[0]
                            if bbox[1]+bbox[3]>height:
                                bbox[3]=height-bbox[1]
                            bbox[0]=int(float(bbox[0])*scalefactor)
                            bbox[1]=int(float(bbox[1])*scalefactor)
                            bbox[2]=int(float(bbox[2])*scalefactor)
                            bbox[3]=int(float(bbox[3])*scalefactor)
                            if (bbox[2]>args.min_anno_size and bbox[3]>args.min_anno_size):
                                # valid annotation found - add to list
                                found_annotations+=1
                                newanno={}
                                newanno["area"]=bbox[2]*bbox[3]
                                newanno["bbox"]=bbox
                                newanno["category_id"]=anno['class']['id']
                                newanno["id"]=annoid
                                newanno["ignore"]=0
                                newanno["image_id"]=cid
                                newanno["iscrowd"]=0
                                newanno["segmentation"]=[]
                                da[dset]['annotations'].append(newanno)
                                annoid+=1
            #found and added annotations
            image=loadImage(frame['frame_folder']+'/Annotations/'+frame['imagePath'])
            newimage=scale(crop(image,[offsetx,offsety,width,height]),[args.width,args.height])
            imagefilename=('%06d.jpg'%cid)
            imagepath=args.destination_folder+'/'+dset+'/'+imagefilename
            saveImage(imagepath,newimage)
            newimagemeta={'file_name':imagefilename,'width':args.width,'height':args.height,'id':cid}
            da[dset]['images'].append(newimagemeta)

        if dset=='train2017' and args.cocofolder!="":
            print("adding coco")
            cocoannos=json.load(open(args.cocofolder+'/annotations/instances_train2017.json'))
            imageannos={}
            for anno in cocoannos['annotations']:
                if anno['image_id'] in imageannos:
                    imageannos[anno['image_id']].append(anno)
                else:
                    imageannos[anno['image_id']]=[anno]
            for img in cocoannos['images']:
                nimg=copy.copy(img)
                ciid=img["id"]
                cid+=1
                if ciid in imageannos:
                    for anno in imageannos[ciid]:
                        newanno=copy.copy(anno)
                        newanno["image_id"]=cid
                        newanno["id"]=annoid
                        da[dset]['annotations'].append(newanno)
                        annoid+=1
                oimagefilename=args.cocofolder+'/train2017/'+img['file_name']
                imagefilename=('%06d.jpg'%cid)
                nimg['file_name']=imagefilename
                nimg['id']=cid
                imagepath=args.destination_folder+'/'+dset+'/'+imagefilename
                shutil.copyfile(oimagefilename,imagepath)
                da[dset]['images'].append(nimg)

    for dset in da:
        json.dump(da[dset],open(args.destination_folder+'/annotations/instances_'+dset+'.json','w'),indent=2)
    print("done")

if __name__ == '__main__':
    main()
