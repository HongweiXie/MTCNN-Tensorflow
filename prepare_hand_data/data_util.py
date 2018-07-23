import os
import numpy as np
import cv2
from prepare_data.utils import IoU
import glob
from pascal_voc_io import PascalVocReader
import tqdm

def landmark_to_bbox(landmark):
    xmin=landmark[0][0]
    ymin=landmark[0][1]
    xmax=landmark[0][0]
    ymax=landmark[0][1]
    for lm in landmark:
        xmin=min(xmin,lm[0])
        ymin=min(ymin,lm[1])
        xmax=max(lm[0],xmax)
        ymax=max(lm[1],ymax)
    # w=xmax-xmin
    # h=ymax-ymin
    # xmin+=w*0.1
    # xmax-=w*0.1
    # ymin+=h*0.1
    # ymax-=h*0.1
    return (int(xmin),int(ymin),int(xmax),int(ymax))

def findInnerBBox(x1, y1, x2, y2,landmark_bboxes):
    if landmark_bboxes is None:
        return None
    for bbox in landmark_bboxes:
        if bbox[0]>=x1 and bbox[2]<=x2 and bbox[1]>=y1 and bbox[3]<=y2:
            return bbox
    return None

def read_voc_annotations(input_dirs):
    data = dict()
    images = []
    bboxes = []
    data['landmarks']=None
    landmark_dict = {}
    for input_dir in input_dirs:
        print('read_annotation: ',input_dir)
        landmark_txt = os.path.join(input_dir, 'landmark.txt')

        if os.path.exists(landmark_txt):
            f = open(landmark_txt, 'r')
            lines = f.readlines()
            for line in lines:
                words = line.split()
                if landmark_dict.get(words[0]) is None:
                    landmark_dict[words[0]] = []
                landmark = np.array(list(map(float, words[5:]))).reshape(-1, 2)
                landmark_dict[words[0]].append(landmark_to_bbox(landmark))
            data['landmarks']=landmark_dict
        else:
            exit(1)
        xml_list=sorted(glob.glob(input_dir + "/*.xml"))
        for xml_file in tqdm.tqdm(xml_list):
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
                if label.startswith('index') or label == 'point_l' or label == 'point_r' or label=='point':
                    bbox.append([xmin, ymin, xmax, ymax])
            bboxes.append(bbox)
            images.append(im_path)

    data['images'] = images  # all image pathes
    data['bboxes'] = bboxes  # all image bboxes
    return data