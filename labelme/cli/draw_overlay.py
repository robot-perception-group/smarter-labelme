#!/usr/bin/env python

import argparse
import base64
import json
import os
import sys

import imgviz

from labelme import utils


PY2 = sys.version_info[0] == 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    json_file = args.json_file
    output_file = args.output_file

    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'': 0}
    label_texts={0:''}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        label_text=label_name
        if shape['flags'] is not None:
            for key in shape['flags']:
                if shape['flags'][key]:
                    label_text+=" {"+key+"}"
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
        label_texts[label_value]=label_text
    lbl, _ = utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = label_texts[value]
    lbl_viz = imgviz.label2rgb(
        label=lbl,
        image=img,
        alpha=0.33,
        label_names=label_names,
        font_size=30,
        loc='rb',
    )
    imgviz.io.imsave(output_file,lbl_viz)
    


if __name__ == '__main__':
    main()
