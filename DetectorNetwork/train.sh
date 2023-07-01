#!/bin/bash

dataset="$1"
snapshotfolder="$2"

if [ -z "$dataset" ]; then
	echo "Usage: $0 <dataset> <snapshotfolder>"
	echo "Will train detector network on <dataset>, saving resulting networks in <snapshotfolder>"
	exit
fi

dataset="$( realpath $dataset )"
snapshotfolder="$( realpath $snapshotfolder )"
mkdir -p $snapshotfolder
cd SSD
python3 main.py --backbone resnet34 --data "$dataset" --mode training --save "$snapshotfolder" --warmup 300 --bs 64
cd ..
