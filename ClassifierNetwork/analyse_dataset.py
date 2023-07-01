#!/usr/bin/env python3
import argparse
import os
import re
import json

def add(array,key,num):
    if key in array:
        array[key]+=num
    else:
        array[key]=num

def main():
    parser = argparse.ArgumentParser(description='Analyse SmarterLabelme Dataset')
    parser.add_argument('folder', type=str, nargs='+', help='folder to analyse')
    args = parser.parse_args()

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
    ids={}
    annotations={}
    classes={}
    flags={}

    for o in filelists:
        for f in filelists[o]:
            frame_id=int(re.search("([0-9]*)\.json",f)[1])
            jdata=json.load(open(o+'/Annotations/'+f))
            annotations[frame_id]=jdata
            for shape in jdata['shapes']:
                add(ids,shape["label"],1)
                res=re.search("(.*)_[0-9]*$",shape["label"])
                if res:
                    tp=str(res[1])
                    add(classes,tp,1)
                    for flag in shape['flags']:
                        if shape['flags'][flag]:
                            add(flags,flag,1)

    print("Classes:")
    print(classes)
    print("Instances:")
    print(ids)
    print("Flags:")
    print(flags)

if __name__ == '__main__':
    main()
