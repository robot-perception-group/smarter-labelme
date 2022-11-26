<h1 align="center">
  <img src="labelme/icons/icon.png"><br/>Smart LabelMe
</h1>

<h4 align="center">
  Video / Image Annotation (Polygon, Semantic mask, Classification) with Python and Re3 and SSD Multibox
</h4>

<br/>

<div align="center">
  <img src="resources/SemanticSegmentation.png" width="70%">
</div>

## Description

Smart-Labelme is a graphical image annotation tool for various image annotation needs such as classification, semantic segmentation, polygonal rois etc.  
It support some smart features like annotation tracking, auto contouring etc. to speed up annotation task.
It is written in Python and uses Qt for its graphical interface.

<i>Auto contouring feature using ~OpenCV grab cut~ Re3 + SSD Multibox</i>
<img src="resources/AutoContour.gif" width="70%" />   

<i>Auto tracking of polygons between frames</i>
<img src="resources/Tracking.gif" width="70%" />   


## Features

- [x] Image annotation for polygon, rectangle, circle, line and point.
- [x] Image flag annotation for classification and cleaning.
- [x] Auto-contouring for fast polygon annotation.
- [x] Auto tracking to track and copy polygon annotations between frames.
- [x] Scripts for semantic segmentation creation from polygonal annotations.
- [x] Video annotation. 
- [x] GUI customization (predefined labels / flags, auto-saving, label validation, etc).
- [x] Exporting VOC-format dataset for semantic/instance segmentation.
- [x] Exporting COCO-format dataset for instance segmentation.


## Requirements

- Ubuntu / macOS / Windows
- Python2 / Python3
- [PyQt4 / PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro) / [PySide2](https://wiki.qt.io/PySide2_GettingStarted)
- PyTorch
- Weights from https://drive.google.com/file/d/17vr3iazbcnSS_ZndbgAAz1mCxg9lJy5f/view?usp=sharing


## Installation
Download the source code onto your local system.
Build package using python setup tool.
Install the package on your system using pip.

```bash
git clone https://github.com/bhavyaajani/smart-labelme
cd smart-labelme
python setup.py build
pip install .
```

### Hint on Pytorch

Pytorch will be installed by pip as a dependency by the above command, if it is not already installed, however you will want to select the matching version for your system from https://pytorch.org/get-started/locally/ -- if you do not have a GPU, use

```
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run `smart_labelme --help` for detail.  
The annotations are saved as a [JSON](http://www.json.org/) file.

```bash
smart_labelme  # just open gui
```
### Important - Get the Re3 weights!

For the Auto-tracking to work, the 300MB weight file (checkpoint.pth) must be located in the current working directory, when smart_labelme is executed.
Download the weights from [Google Drive](https://drive.google.com/file/d/17vr3iazbcnSS_ZndbgAAz1mCxg9lJy5f/view?usp=sharing) and unzip them in the current folder before running smart_labelme.

The same is true for the SSD Multibox weights. The file is currently called "ssd.pt" and must be in the same folder as checkpoint.pth, the current working directory when smart_labelme is started.

### Command Line Arguments
- `--output` specifies the location that annotations will be written to. Annotations will be stored in this directory with a name that corresponds to the image that the annotation was made on.
- The first time you run labelme, it will create a config file in `~/.labelmerc`. You can edit this file and the changes will be applied the next time that you launch labelme. If you would prefer to use a config file from another location, you can specify this file with the `--config` flag.
- Without the `--nosortlabels` flag, the program will list labels in alphabetical order. When the program is run with this flag, it will display labels in the order that they are provided.
- Flags are assigned to an entire image. 
- Labels are assigned to a single polygon.

## Acknowledgement

This repo is the fork of [bhavyaajani/smart-labelme](https://github.com/bhavyaajani/smart-labelme).


## Cite This Project

If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entry.

```bash
@misc{smart-labelme2020,
  author =       {Bhavya Ajani},
  title =        {{Smart-labelme: Video / Image Annotation (Polygon, Semantic mask, Classification) with Python}},
  howpublished = {\url{https://github.com/bhavyaajani/smart-labelme}},
  year =         {2020}
}
```

```bash
@misc{labelme2016,
  author =       {Kentaro Wada},
  title =        {{labelme: Image Polygonal Annotation with Python}},
  howpublished = {\url{https://github.com/wkentaro/labelme}},
  year =         {2016}
}
```
