#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np

test_mode = "onet"
thresh = [0.6, 0.4, 0.5]
min_face_size = 80
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_hand/Hand_PNet24_landmark_16_64_2/PNet', '../data/MTCNN_model/Hand_RNet_landmark/RNet', '../data/MTCNN_model/Hand_ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
# RNet = Detector(R_Net, 24, 1, model_path[1])
# detectors[1] = RNet
# ONet = Detector(O_Net, 48, 1, model_path[2])
# detectors[2] = ONet
videopath = "./test_input.avi"
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('result_landmark/output_pnet.avi', fourcc, 20.0, (640, 480))
video_capture = cv2.VideoCapture(videopath)
video_capture.set(3, 640)
video_capture.set(4, 480)
corpbbox = None
while True:
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = video_capture.read()
    if ret:
        image = frame
        all_boxes, boxes_c, _ = mtcnn_detector.detect_pnet(image)

        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        if (boxes_c is not None):
            for bbox in boxes_c:
                cv2.putText(image, str(np.round(bbox[4], 2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0))
            cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
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
