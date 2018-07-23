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
    cx=(bbox[0]+bbox[2])/2
    cy=(bbox[1]+bbox[3])/2
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    left=max(0,cx-(w*0.5))
    top=max(0,cy-(h*0.5))
    right=cx+w*0.5
    bottom=cy+h*0.5
    return dlib.rectangle(int(left),int(top),int(right),int(bottom))


predictor_path='/home/sixd-ailabs/Develop/Human/Hand/Code/build-Hand-Landmarks-Detector-Desktop_Qt_5_10_0_GCC_64bit-Default/Hand_9_Landmarks_Detector.dat'
predictor = dlib.shape_predictor(predictor_path)

test_mode = "onet"
thresh = [0.6, 0.5, 0.3]
min_face_size = 60
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_hand/Hand_PNet24_landmark_16_64_2/PNet', '../data/MTCNN_hand/Hand_RNet_landmark_2/RNet', '../data/MTCNN_hand/Hand_ONet_landmark/ONet']
epoch = [18, 18, 20]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
# ONet = Detector(O_Net, 48, 1, model_path[2])
# detectors[2] = ONet
videopath = "./test_input.avi"
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('result_landmark/output_rnet.avi', fourcc, 20.0, (640, 480))
video_capture = cv2.VideoCapture(videopath)
video_capture.set(3, 640)
video_capture.set(4, 480)
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
            # shape = predictor(image, convert2dlibbbox(bbox))
            # for i in range(5):
            #     pt = shape.part(i)
            #     cv2.circle(frame, (int(pt.x), int(pt.y)), 5, (55, 255, 155),2)
            #     print(shape.part(i))
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (0, 255, 0), 1)
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
        if landmarks is not None:
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i])/2):
                    cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 5, (255, 255, 0),2)
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
