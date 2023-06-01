#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('video', type=str, help='input video to be converted to frames')
    parser.add_argument('output_dir', type=str, help='output directory where frames to be saved')
    parser.add_argument('--fps', type=str, help='Framerate at which to extract, Default: 8 - use "full" for original video framerate', default="8")


    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    if(args.fps=="full"):
        command="ffmpeg -i '"+args.video+"' -q:v 0 -f image2 -frame_pts true '"+osp.join(args.output_dir,'frame_%06d.jpg')+"'"
    else:
        try:
            fps=float(args.fps)
        except:
            print('--fps must be "full" or a valid floating point number')
            sys.exit(1)
        command="ffmpeg -i '"+args.video+"' -filter:v 'fps="+args.fps+":round=up' -q:v 0 -f image2 -frame_pts true '"+osp.join(args.output_dir,'frame_%06d.jpg')+"'"

    # check if ffmpeg is present
    result=subprocess.call("ffmpeg -version", shell=True)
    if result!=0:
        print("Error running ffmpeg - please make sure ffmpeg is installed on your system!")
        sys.exit(1)

    try:
        os.makedirs(args.output_dir)
    except:
        print('Cannot create output directory:', args.output_dir)
        sys.exit(1)
    
    print('Extracting frames to :', args.output_dir)

    result=subprocess.call(command,shell=True)
    if result!=0:
        print("Subprocess ffmpeg returned error. Please check above error message.")
        sys.exit(result)
    print('Successfully extracted video '+args.video+' to '+args.output_dir+' with '+args.fps+' fps.')

if __name__ == '__main__':
    main()
