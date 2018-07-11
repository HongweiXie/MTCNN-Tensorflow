#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from prepare_data.utils import IoU
import glob
from pascal_voc_io import PascalVocReader
sys.path.append("/home/sixd-ailabs/Develop/Human/DatasetTools/dianjin")
import greenscreen as gs

input_dir = "/home/sixd-ailabs/Develop/Human/Hand/diandu/youeryuan_lenovo"
background_dir='/home/sixd-ailabs/Develop/Human/Hand/diandu/background'
save_dir = "/home/sixd-ailabs/Develop/Human/Hand/diandu/Train/12"
pos_save_dir = os.path.join(save_dir,'positive')
part_save_dir = os.path.join(save_dir,'part')
neg_save_dir = os.path.join(save_dir,'negative')

net_size=24
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'g_pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'g_neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'g_part_12.txt'), 'w')


min_hand=80

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

background_images=[]
for jpg_file in glob.glob(background_dir+"/*/*.jpg"):
    img=cv2.imread(jpg_file)
    background_images.append(img)
    height, width, channel = img.shape
    neg_num = 0
    # 1---->50
    while neg_num < 30:
        # neg_num's size [40,min(width, height) / 2],min_size:40
        size = npr.randint(net_size, min(width, height) / 2)
        # top_left
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        # random crop
        crop_box = np.array([nx, ny, nx + size, ny + size])

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)

        # Iou with all gts must below 0.3
        save_file = os.path.join(neg_save_dir, "g_%s.jpg" % n_idx)
        f2.write("12/negative/g_%s.jpg" % n_idx + ' 0\n')
        cv2.imwrite(save_file, resized_im)
        n_idx += 1
        neg_num += 1

background_num=len(background_images)

for xml_file in glob.glob(input_dir+"/*.xml"):
    print os.path.basename(xml_file)
    reader=PascalVocReader(xml_file)
    #image path
    im_path = xml_file[:-4]+".jpg"
    shapes = reader.getShapes()
    bbox=[]
    bbox_other=[]
    for shape in shapes:
        points = shape[1]
        xmin = points[0][0]
        ymin = points[0][1]
        xmax = points[2][0]
        ymax = points[2][1]
        label = shape[0]
        if label=='index' or label=='point_l' or label=='point_r':
            bbox.append(xmin)
            bbox.append(ymin)
            bbox.append(xmax)
            bbox.append(ymax)
        else:
            bbox_other.append(xmin)
            bbox_other.append(ymin)
            bbox_other.append(xmax)
            bbox_other.append(ymax)

    #gt
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    other_boxes=np.array(bbox_other, dtype=np.float32).reshape(-1, 4)
    #load image
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"
        
    height, width, channel = img.shape
    #as for neg
    for box in other_boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1

        # print x1,y1,x2,y2,w,h

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < min_hand or x1 < 0 or y1 < 0 or min(h,w)<net_size:
            continue
                # generate positive examples and part faces
        for i in range(10):
            # pos and part face size [minsize*0.8,maxsize*1.25]
            size = npr.randint(int(min(w, h) * 0.9), np.ceil(1.2 * max(w, h)))
            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.1, w * 0.1)
            delta_y = npr.randint(-h * 0.1, h * 0.1)
            # show this way: nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            # show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            # yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            # crop
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            bgIndex=npr.randint(0,background_num-1)
            # cv2.imshow("bg",background_images[bgIndex])
            merge_im=gs.mergeImage(cropped_im,background_images[bgIndex])
            # cv2.imshow("out",merge_im/255)
            # cv2.waitKey(0)
            # resize
            resized_im = cv2.resize(merge_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)
            Iou = IoU(crop_box, boxes)
            if len(Iou)<=0 or np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "g_%s.jpg" % n_idx)
                f2.write("12/negative/g_%s.jpg" % n_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                box_ = box.reshape(1, -1)
                n_idx += 1
        box_idx += 1

    #as for 正 part样本
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        #gt's width
        w = x2 - x1 + 1
        #gt's height
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        for i in range(5):
            size = npr.randint(net_size, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
    
            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            merge_im = gs.mergeImage(cropped_im, background_images[npr.randint(0, background_num - 1)])

            resized_im = cv2.resize(merge_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "g_%s.jpg" % n_idx)
                f2.write("12/negative/g_%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                # cv2.imshow("neg", merge_im / 255)
                # cv2.waitKey(0)
                n_idx += 1        
	# generate positive examples and part faces
        for i in range(20):
            # pos and part face size [minsize*0.8,maxsize*1.25]
            size = npr.randint(int(min(w, h) * 0.9), np.ceil(1.2 * max(w, h)))

            delta=0.1
            # delta here is the offset of box center
            delta_x = npr.randint(-w * delta, w * delta)
            delta_y = npr.randint(-h * delta, h * delta)
            #show this way: nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            #show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue 
            crop_box = np.array([nx1, ny1, nx2, ny2])
            #yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            #crop
            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            merge_im = gs.mergeImage(cropped_im, background_images[npr.randint(0, background_num - 1)])
            # cv2.imshow("pos", merge_im/255)
            # cv2.waitKey(0)
            #resize
            resized_im = cv2.resize(merge_im, (net_size, net_size), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "g_%s.jpg"%p_idx)
                f1.write("12/positive/g_%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "g_%s.jpg"%d_idx)
                f3.write("12/part/g_%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
    print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)
f1.close()
f2.close()
f3.close()
