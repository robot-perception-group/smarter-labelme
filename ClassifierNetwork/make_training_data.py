#!/usr/bin/env python3
import argparse
import os
import re
import json
import numpy as np
import torch
import torchvision

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
    if w>h:
        cw=int(w/2)+1
    else:
        cw=int(h/2)+1
    l=(x-int(w/2))+cw
    r=l+w
    t=(y-int(h/2))+cw
    b=t+h
    p=torchvision.transforms.Pad(cw,padding_mode='reflect')
    pimage=p(image)
    return pimage[:,t:b,l:r]

def main():
    parser = argparse.ArgumentParser(description='Create Behaviour Dataset from SmarterLabelme dataset')
    parser.add_argument('folder', type=str, nargs='+', help='folder to analyse')
    parser.add_argument('destination_folder', type=str, help='folder to store images and json')
    parser.add_argument('--flags', type=str, help='dictionary of interesting behaviors in dataset and what they should be called',default='{"grazing":"grazing","standing":"standing","walking":"walking","running":"running"}')
    parser.add_argument('--exclude', type=str, help='list of behaviors to exclude from dataset if tagged',default='["auto_flag"]')
    parser.add_argument('--width', type=int, help='dimensions for image', default=300)
    parser.add_argument('--height', type=int, help='dimensions for image', default=300)
    parser.add_argument('--min_size', type=float, help='width or height of image taken by annotation', default=0.6)
    parser.add_argument('--max_size', type=float, help='width or height of image taken by annotation', default=0.8)
    parser.add_argument('--trainsize', type=int, help='how much data to create', default=7000)
    parser.add_argument('--valsize', type=int, help='how much data to create', default=1000)
    parser.add_argument('--testsize', type=int, help='how much data to create', default=2000)
    parser.add_argument('--unknown', type=int, help='whether to include random crops as unknown class', default=1)
    parser.add_argument('--random', type=int, help='random seed for dataset generation', default=1337)
    parser.add_argument('--test-percentage', type=float, help='Percentage of data reserved for test set', default=20)
    parser.add_argument('--val-percentage', type=float, help='Percentage of data reserved for test set', default=10)
    parser.add_argument('--test-type', type=str, help='How to test random or fixed or by id - fixed means the beginning of each set is train and the second part is val followed by test, id splits based on annotation id', default='id')

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
        args_flags=json.loads(args.flags)
    except Exception as e:
        print("Error parsing flags - invalid json string: %s - %s"%(args.flags,e))
        exit(1)
    if not isinstance(args_flags,dict):
        print("Error parsing flags - %s is not a dictionary"%args.flags)
        exit(1)
    try:
        args_exclude=json.loads(args.exclude)
    except Exception as e:
        print("Error parsing exclude flags - invalid json string: %s - %s"%(args.exclude,e))
        exit(1)
    if not isinstance(args_exclude,list):
        print("Error parsing flags - %s is not a list"%args.exclude)
        exit(1)


    target_behaviours={}
    for flag in args_flags:
        if args_flags[flag] not in target_behaviours:
            target_behaviours[args_flags[flag]]={'source':[],'train':[],'val':[],'test':[]}
        target_behaviours[args_flags[flag]]['source'].append(flag)

    ids={}
    annotations={'train':{},'val':{},'test':{}}
    classes={}
    flags={}
    setflags={'train':{},'val':{},'test':{}}
    excluded=0

    idset={}
    if args.test_type=='id':
        labelarray=[]
        for o in filelists:
            for f in filelists[o]:
                frame_id=int(re.search("([0-9]*)\.json",f)[1])
                jdata=json.load(open(o+'/Annotations/'+f))
                jdata['frame_id']=frame_id
                jdata['frame_file']=f
                jdata['frame_folder']=o
                for shape in jdata['shapes']:
                    if str(shape["label"]) not in labelarray:
                        labelarray.append(str(shape["label"]))
                #annotations are used for unknown data only - in this case pull from all frames and rely on rng
                annotations['train'][frame_id]=jdata
                annotations['test'][frame_id]=jdata
                annotations['val'][frame_id]=jdata

        #randomly assign each label to train, val or test
        rng.shuffle(labelarray)
        for i,j in enumerate(labelarray):
            r=i*100/len(labelarray)
            if r>=(100-args.test_percentage):
                idset[j]='test'
            elif r>=(100-(args.test_percentage+args.val_percentage)):
                idset[j]='val'
            else:
                idset[j]='train'
            print("Assigning... %s in %s"%(j,idset[j]))

    for o in filelists:
        fid=0

        for f in filelists[o]:
            frame_id=int(re.search("([0-9]*)\.json",f)[1])
            jdata=json.load(open(o+'/Annotations/'+f))
            dset='train'
            r=0
            if args.test_type=='random':
                r=rng.random()*100
            elif args.test_type=='fixed':
                r=(100*fid/len(filelists[o]))
            if r>=(100-args.test_percentage):
                dset='test'
            elif r>=(100-(args.test_percentage+args.val_percentage)):
                dset='val'
            if args.test_type!='id':
                annotations[dset][frame_id]=jdata
            jdata['frame_id']=frame_id
            jdata['frame_file']=f
            jdata['frame_folder']=o
            for shape in jdata['shapes']:
                if args.test_type=='id':
                    dset=idset[str(shape["label"])]
                add(ids,shape["label"],1)
                res=re.search("(.*)_[0-9]*$",shape["label"])
                if res:
                    tp=str(res[1])
                    add(classes,tp,1)
                    if len(shape['flags']) and not max([ (flag in args_exclude and shape['flags'][flag]) for flag in shape['flags']]):
                        for flag in shape['flags']:
                            if shape['flags'][flag]:
                                if flag in args_flags:
                                    anno={  'json':jdata,
                                            'shape':shape}
                                    target_behaviours[args_flags[flag]][dset].append(anno)
                                add(flags,flag,1)
                                add(setflags[dset],flag,1)
                    else:
                        excluded+=1
            fid+=1

    print("Classes:")
    print(json.dumps(classes,indent=1))
    print("Instances:")
    print(json.dumps(ids,indent=1))
    print("Flags:")
    print(json.dumps(flags,indent=1))
    print("Assigned to train:");
    print(json.dumps(setflags['train'],indent=1))
    print("Assigned to val:");
    print(json.dumps(setflags['val'],indent=1))
    print("Assigned to test:");
    print(json.dumps(setflags['test'],indent=1))
    print("Result Classes:")
    print(json.dumps(list(target_behaviours.keys()),indent=1))
    if args.unknown:
        print('+ ["unknown"]')
    print("Excluded Annotations: %d"%(excluded))

    da={'train':{'classes':[],'images':[]},'val':{'classes':[],'images':[]},'test':{'classes':[],'images':[]}}
    try:
        for dset in da:
            os.mkdir(args.destination_folder+'/'+dset)
            os.mkdir(args.destination_folder+'/'+dset+'/Images')
    except:
        pass

    if args.unknown:
        print("creating unknown annotations ...")
        for dset in da:
            da[dset]['classes'].append('unknown')
            s=args.trainsize
            if dset=='test':
                s=args.testsize
            if dset=='val':
                s=args.valsize
            for i in range(s):
                frame=rng.choice(list(annotations[dset].keys()))
                anno=annotations[dset][frame]
                x=rng.integers(int(anno['imageWidth']*0.8))+int(0.1*anno['imageWidth'])
                y=rng.integers(int(anno['imageHeight']*0.8))+int(0.1*anno['imageHeight'])
                w=rng.integers(low=int(anno['imageWidth']*0.1),high=int(anno['imageWidth']*0.5))
                h=w
                image=loadImage(anno['frame_folder']+'/Annotations/'+anno['imagePath'])
                newimage=scale(crop(image,[x,y,w,h]),[args.width,args.height])
                cid=len(da[dset]['images'])
                imagefilename=('Images/%06d.jpg'%cid)
                imagepath=args.destination_folder+'/'+dset+'/'+imagefilename
                saveImage(imagepath,newimage)
                newanno={'image':imagefilename,'width':args.width,'height':args.height,'class':'unknown'}
                da[dset]['images'].append(newanno)

    
    for b in target_behaviours:
        print("creating %s annotations ..."%b)
        for dset in da:
            da[dset]['classes'].append(b)
            if len(target_behaviours[b][dset]):
                s=args.trainsize
                if dset=='test':
                    s=args.testsize
                if dset=='val':
                    s=args.valsize
                for i in range(s):
                    anno=rng.choice(target_behaviours[b][dset])
                    W=anno['json']['imageWidth']
                    H=anno['json']['imageHeight']
                    x1=min(max(anno['shape']['points'][0][0],0),W-1)
                    x2=min(max(anno['shape']['points'][1][0],0),W-1)
                    y1=min(max(anno['shape']['points'][0][1],0),H-1)
                    y2=min(max(anno['shape']['points'][1][1],0),H-1)
                    f=(rng.random()*((1.0/args.min_size) - (1.0/args.max_size)))+(1.0/args.max_size)
                    x=int((x1+x2)/2)
                    y=int((y1+y2)/2)
                    w=max(int(abs((x2-x1)*f)),1)
                    h=max(int(abs((y2-y1)*f)),1)
                    if (w>h):
                        h=w
                    else:
                        w=h
                    image=loadImage(anno['json']['frame_folder']+'/Annotations/'+anno['json']['imagePath'])
                    newimage=scale(crop(image,[x,y,w,h]),[args.width,args.height])
                    cid=len(da[dset]['images'])
                    imagefilename=('Images/%06d.jpg'%cid)
                    imagepath=args.destination_folder+'/'+dset+'/'+imagefilename
                    saveImage(imagepath,newimage)
                    newanno={'image':imagefilename,'width':args.width,'height':args.height,'class':b}
                    da[dset]['images'].append(newanno)

    for dset in da:
        json.dump(da[dset],open(args.destination_folder+'/'+dset+'/annotations.json','w'),indent=2)


if __name__ == '__main__':
    main()
