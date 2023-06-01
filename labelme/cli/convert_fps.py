#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys

def main():
    parser = argparse.ArgumentParser(
            description="changes fps on single frame annotated video (changes frame numbers)",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_annotation_folder')
    parser.add_argument('dest_annotation_folder')
    parser.add_argument('--source_framerate','-sf',required=True,type=float,help="Framerate of the source annotations")
    parser.add_argument('--dest_framerate','-df',default=30000./1001,type=float,help="Resample to this framerate")
    parser.add_argument('--dest_template',default='frame_%05d',help="Frame file name template for resulting annotations")
    parser.add_argument('--dest_image_type',default='.jpg',help="Image file name extension written in resulting annotation")
    args = parser.parse_args()
    if not os.path.isdir(args.source_annotation_folder):
        print("Error: source annotation folder %s cannot be opened\n"%args.source_annotation_folder)
        exit(1)
    if not os.path.isdir(args.dest_annotation_folder):
        print("Error: destination annotation folder %s cannot be opened\n"%args.dest_annotation_folder)
        exit(1)
    try:
        filelist=sorted([f for f in os.listdir(args.source_annotation_folder) if (re.search("\.json$",f) is not None)])
    except:
        print("Error: Could not get annotations from %s\n"%args.source_annotation_folder)
        exit(1)

    for f in filelist:
        frame_id=int(re.search("([0-9]*)\.json",f)[1])
        new_frame_id=int(frame_id*args.dest_framerate/args.source_framerate)
        df=args.dest_annotation_folder+'/'+(args.dest_template%new_frame_id)+'.json'
        if not os.path.isfile(df):
            jdata=json.load(open(args.source_annotation_folder+'/'+f))
            jdata['originalImagePath']=jdata['imagePath']
            jdata['imagePath']="../"+(args.dest_template%new_frame_id)+args.dest_image_type
            json.dump(jdata,open(df,'w'),ensure_ascii=False,indent=2)
        
    exit(0)
if __name__ == '__main__':
    main()
