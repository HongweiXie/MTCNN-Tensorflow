#coding:utf-8
import sys
import glob
import tqdm
import os
from pascal_voc_io import PascalVocWriter
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np
import dlib

input_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/chengren_17'
output_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_chengren_17_lr'


def convert2dlibbbox(bbox):
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    left=max(0,cx-(w*0.5))
    top=max(0,cy-(h*0.5))
    right=cx+w*0.5
    bottom=cy+h*0.5
    return dlib.rectangle(int(left),int(top),int(right),int(bottom))


predictor_path='/home/sixd-ailabs/Develop/Human/Hand/Code/build-Hand-Landmarks-Detector-Desktop_Qt_5_10_0_GCC_64bit-Default/Hand_5_Landmarks_Detector_r_aug.dat'
predictor = dlib.shape_predictor(predictor_path)

test_mode = "onet"
thresh = [0.8, 0.5, 0.6]
min_face_size = 150
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_hand/Hand_PNet24_landmark_16_64_3/PNet', '../data/MTCNN_hand/Hand_RNet_landmark_3/RNet', '../data/MTCNN_hand/Hand_ONet_landmark_3/ONet']
epoch = [18, 20, 20]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
videopath = "./test_input.avi"
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# Define the codec and create VideoWriter object
corpbbox = None

jpg_list=glob.glob(input_path+'/*.jpg')
for jpg_file in tqdm.tqdm(jpg_list):
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    frame=cv2.imread(jpg_file)
    image = np.array(frame)
    boxes_c, landmarks = mtcnn_detector.detect(image)

    image_name = os.path.basename(jpg_file)
    xml_name = image_name[:-4] + ".xml"
    writer = PascalVocWriter('test', image_name, imgSize=image.shape,
                             localImgPath=os.path.join(output_path, image_name))

    # print landmarks.shape
    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()
    fps = 1.0 / t
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        writer.addBndBox(corpbbox[0],corpbbox[1],corpbbox[2],corpbbox[3],'index',False)
        shape = predictor(image, convert2dlibbbox(bbox))
        for i in range(5):
            pt = shape.part(i)
            cv2.circle(frame, (int(pt.x), int(pt.y)), 5, (55, 255, 155), 2)
            print(shape.part(i))
        # if score > thresh:
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (0, 255, 0), 1)
        cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
    cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 255), 2)
    colors = [(55, 255, 155), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 0, 0)]
    # for i in range(landmarks.shape[0]):
    #     for j in range(len(landmarks[i])/2):
    #         cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 5, colors[j],2)
    # time end
    cv2.imshow("", frame)
    writer.save(os.path.join(output_path,xml_name))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
