# coding: utf-8
import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
from prepare_data.BBox_utils import getDataFromTxt,processImage,shuffle_in_unison_scary,BBox
from prepare_data.Landmark_utils import show_landmark,rotate,flip
import random
import tensorflow as tf
import sys
import numpy.random as npr
import data_util

OUTPUT = '/home/sixd-ailabs/Develop/Human/Hand/diandu/Train/24'
dstdir = os.path.join(OUTPUT,"train_PNet_landmark_aug")
if not exists(OUTPUT): os.mkdir(OUTPUT)
if not exists(dstdir): os.mkdir(dstdir)
assert(exists(dstdir) and exists(OUTPUT))

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
     # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr
def GenerateData(ftxts, output,net,argument=False):
    if net == "PNet":
        size = 24
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print 'Net type error'
        return
    image_id = 0
    f = open(join(OUTPUT,"landmark_%s.txt" %(size)),'w')
    #dstdir = "train_landmark_few"
    data=[]
    for ftxt in ftxts:
        tmp=getDataFromTxt(ftxt)
        data+=tmp
    idx = 0
    #image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        #print imgPath
        F_imgs = []
        F_landmarks = []        
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        f_face = cv2.resize(f_face,(size,size))
        landmark = np.zeros((5, 2))
        #normalize
        for index, one in enumerate(landmarkGt):
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))        
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print idx, "images done"
            x1, y1, x2, y2 = gt_box

            landmark_inner_bbox=data_util.landmark_to_bbox(landmarkGt)
            #gt's width
            gt_w = x2 - x1 + 1
            #gt's height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 80 or x1 < 0 or y1 < 0:
                continue
            #random shift
            for i in range(5):
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                # delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                # nx1 = max(x1+gt_w/2-bbox_size/2+delta_x,0)
                # ny1 = max(y1+gt_h/2-bbox_size/2+delta_y,0)
                #
                # nx2 = nx1 + bbox_size
                # ny2 = ny1 + bbox_size

                offset_bias = 0
                inner_xmin, inner_ymin, inner_xmax, inner_ymax = landmark_inner_bbox
                inner_w = inner_xmax - inner_xmin + 1
                inner_h = inner_ymax - inner_ymin + 1
                bbox_size = max(bbox_size, inner_h + 1, inner_w + 1) + offset_bias
                nx1 = npr.randint(max(0, inner_xmax - bbox_size, x1 - 0.2 * gt_w), max(1, inner_xmin - offset_bias, x1))
                ny1 = npr.randint(max(0, inner_ymax - bbox_size, y1 - 0.2 * gt_h), max(1, inner_ymin - offset_bias, y1))
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size

                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    #normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))

                    # print('resized_im')
                    # show_landmark(resized_im,landmark)
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)
                    bbox = BBox([nx1,ny1,nx2,ny2])                    

                    #mirror                    
                    if random.choice([0,1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        # print('face_flipped')
                        # show_landmark(face_flipped,landmark_flipped)
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    #rotate
                    for i in range(1):
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), npr.randint(-30,30))#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))

                        # print('face_rotated_by_alpha')
                        # show_landmark(face_rotated_by_alpha,landmark_rotated)
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))

                        # print('face_flip_rotate')
                        # show_landmark(face_flipped,landmark_flipped)
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    # #inverse clockwise rotation
                    # if random.choice([0,1]) > 0:
                    #     face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                    #                                                      bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                    #     landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    #     face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                    #     F_imgs.append(face_rotated_by_alpha)
                    #     F_landmarks.append(landmark_rotated.reshape(10))
                    #
                    #     face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    #     face_flipped = cv2.resize(face_flipped, (size, size))
                    #     F_imgs.append(face_flipped)
                    #     F_landmarks.append(landmark_flipped.reshape(10))
                    
            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
            #print F_imgs.shape
            #print F_landmarks.shape
            for i in range(len(F_imgs)):
                print image_id

                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                cv2.imwrite(join(dstdir,"%d.jpg" %(image_id)), F_imgs[i])
                landmarks = map(str,list(F_landmarks[i]))
                f.write(join(dstdir,"%d.jpg" %(image_id))+" -2 "+" ".join(landmarks)+"\n")
                image_id = image_id + 1
            
    #print F_imgs.shape
    #print F_landmarks.shape
    #F_imgs = processImage(F_imgs)
    #shuffle_in_unison_scary(F_imgs, F_landmarks)
    
    f.close()
    return F_imgs,F_landmarks

if __name__ == '__main__':
    # train data
    net = "RNet"
    #train_txt = "train.txt"
    train_txts = ["/home/sixd-ailabs/Develop/Human/Hand/diandu/chengren_17/landmark.txt",'/home/sixd-ailabs/Develop/Human/Hand/diandu/test/output/landmark.txt','/home/sixd-ailabs/Develop/Human/Hand/diandu/zhijian/youeryuan_dell/landmark.txt']
    imgs,landmarks = GenerateData(train_txts, OUTPUT,net,argument=True)
    
   
