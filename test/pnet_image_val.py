#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
import Util
test_mode = "ONet"
thresh = [0.9]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet']
epoch = [18]
batch_size = [2048]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet


mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

import os
draw_result=False
test_data_dir='/home/sixd-ailabs/Develop/Human/Face/train'
test_file='testImageList.txt'
f=open(os.path.join(test_data_dir,test_file),'r')
lines=f.readlines()
sum_=len(lines)
cnt_true_pos=0
cnt_false_pos=0
cnt=0
for line in lines:
    #imgpath = "test/0_Parade_marchingband_1_353.jpg"
    #imgpath = "input.png"
    words = line.split()
    imgpath=os.path.join(test_data_dir,words[0])
    image = cv2.imread(imgpath)
    all_boxes, all_boxes_calib, _ = mtcnn_detector.detect_pnet(image)
    rectangles = all_boxes_calib
    img_name=os.path.basename(imgpath)
    gt_bbox=(int(words[1]),int(words[3]),int(words[2]),int(words[4]))
    hit_true_pos=False
    if rectangles is not None:
        for rectangle in rectangles:
            rect=(int(rectangle[0]), int(rectangle[1]), int(rectangle[2]), int(rectangle[3]))
            iou=Util.IoU(gt_bbox,rect)
            if(iou>0.4):
                hit_true_pos=True
            else:
                cnt_false_pos+=1
        if hit_true_pos:
            cnt_true_pos+=1

    cnt+=1
    view_bar(cnt,sum_)
    if draw_result:
        img = cv2.imread(imgpath)
        draw = img.copy()
        for rectangle in rectangles:
            cv2.putText(draw, str(rectangle[4]), (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0))
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (0, 255, 0), 1)

        cv2.imshow("test", draw)
        if(cv2.waitKey(0)==27):
            cv2.imwrite('test.jpg', draw)
            cv2.destroyAllWindows()
print '\n'
print cnt_true_pos*1.0/sum_
print cnt_false_pos


'''
for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''