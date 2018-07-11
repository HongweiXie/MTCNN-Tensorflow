import os
import numpy as np
import cv2
from prepare_data.utils import IoU
import glob
from pascal_voc_io import PascalVocReader

def read_voc_annotations(input_dirs):
    data = dict()
    images = []
    bboxes = []
    for input_dir in input_dirs:
        for xml_file in glob.glob(input_dir + "/*.xml"):
            reader = PascalVocReader(xml_file)
            # image path
            im_path = xml_file[:-4] + ".jpg"
            shapes = reader.getShapes()
            bbox = []
            for shape in shapes:
                points = shape[1]
                xmin = points[0][0]
                ymin = points[0][1]
                xmax = points[2][0]
                ymax = points[2][1]
                label = shape[0]
                if label.startswith('index') or label == 'point_l' or label == 'point_r':
                    bbox.append([xmin, ymin, xmax, ymax])
            bboxes.append(bbox)
            images.append(im_path)

    data['images'] = images  # all image pathes
    data['bboxes'] = bboxes  # all image bboxes
    return data