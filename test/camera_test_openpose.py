#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np
import dlib

def convert2dlibbbox(bbox):
    expand_ratio=0.1
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    left=max(0,cx-(w*(0.5+expand_ratio)))
    top=max(0,cy-(h*(0.5+expand_ratio)))
    right=cx+w*(0.5+expand_ratio)
    bottom=cy+h*(0.5+expand_ratio)
    return dlib.rectangle(int(left),int(top),int(right),int(bottom))

def expand_bbox(bbox):
    expand_ratio=0.1
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    left=max(0,cx-(w*(0.5+expand_ratio)))
    top=max(0,cy-(h*(0.5+expand_ratio)))
    right=cx+w*(0.5+expand_ratio)
    bottom=cy+h*(0.5+expand_ratio)
    return [int(left),int(top),int(right),int(bottom)]

# Load openpose.
sys.path.append('/usr/local/python')
#sys.path.append('./python')
from openpose import *

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = "/home/sixd-ailabs/Develop/Human/Hand/Code/openpose/models/"

openpose = OpenPose(params)
predictor_path='/home/sixd-ailabs/Develop/Human/Hand/Code/build-Hand-Landmarks-Detector-Desktop_Qt_5_10_0_GCC_64bit-Default/Hand_5_Landmarks_Detector_100.dat'
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
epoch = [20, 20, 22]
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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('result_landmark/output_onet.avi', fourcc, 20.0, (1280, 720))
video_capture = cv2.VideoCapture(1)
video_capture.set(3, 1280)
video_capture.set(4, 720)
corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()
    # video.write(frame)
    if ret:
        image = np.array(frame)
        boxes_c,landmarks = mtcnn_detector.detect(image)
        
        # print landmarks.shape
        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

            hands_rectangles=[[expand_bbox(bbox),[0,0,0,0]]]
            left_hands, right_hands, _ = openpose.forward_hands(frame, hands_rectangles, True)
            index_finger = left_hands[0][8]
            print(index_finger[2])
            cv2.circle(frame, (int(index_finger[0]), int(index_finger[1])), 5, (0, 0, 255), -1)


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
        video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break
    else:
        print 'device not find'
        break
video_capture.release()
cv2.destroyAllWindows()
