
import cv2
import os
import numpy as np
import sys
import math
sys.path.append('/home/sixd-ailabs/Develop/Human/Hand/Code/build-marker-detect-with-color-pen-c-Desktop_Qt_5_10_0_GCC_64bit-Default')
import ReadBook

time_interval=0.5

class BookReader():
    def __init__(self):
        self.fingerTipList=[]
        self.preReadBookID=-1
        self.preReadPageIndex=-1
        self.preReadLabelID=-1
        self.preReadTimeStamp=0
        self.preAudioPath=''
        self.nativeBookReader=ReadBook.ReadBook \
            ("/home/sixd-ailabs/Develop/Human/Hand/Code/marker-detect-with-color-pen-c/data/2d_recognize/Images",
             "/home/sixd-ailabs/Develop/Human/Hand/Code/marker-detect-with-color-pen-c/data/2d_recognize/book_id_path_tab_config.txt")

    def distance(self,p1, p2):
        x_dis = float(p1[0] - p2[0])
        y_dis = float(p1[1] - p2[1])
        return math.sqrt(x_dis * x_dis + y_dis * y_dis)

    def avgPosition(self,posList):
        x = 0.0
        y = 0.0
        cnt = 0
        for p in posList:
            if(p[0]<0 or p[1]<0):
                continue
            x += p[0]
            y += p[1]
            cnt += 1
        return x / cnt, y / cnt

    def on_detected_finger_tip(self, img, pos):
        x, y, timestamp = pos
        self.fingerTipList.append(pos)
        while len(self.fingerTipList) > 0 and (timestamp - self.fingerTipList[0][2])/cv2.getTickFrequency() > time_interval:
            self.fingerTipList.pop(0)
        num = len(self.fingerTipList)
        # print('tiplist:',num)
        # if num < 5:
        #     return
        valid_num = 0
        for tip in self.fingerTipList:
            if tip[0] >= 0 and tip[1] >= 0:
                valid_num += 1
        if valid_num >= num * 0.8:
            ax, ay = self.avgPosition(self.fingerTipList)
            max_dis = 0
            min_dis = 10000000
            mean_dis=0.0
            for p in self.fingerTipList:
                if p[0]<0 or p[1]<0:
                    continue
                dis = self.distance(p, (ax, ay))
                max_dis = max(max_dis, dis)
                min_dis = min(min_dis, dis)
                mean_dis+=dis
            mean_dis/=valid_num
            # print(max_dis,mean_dis)
            if (max_dis < mean_dis * 2 and max_dis<20):
                image_char = img.astype(np.uint8).tostring()
                print('read:', ax, ay)
                ret=self.nativeBookReader.readPointOnFrame(image_char,img.shape[0], img.shape[1],ax,ay)
                if(ret[0]!=1):
                    print('Error:',ret[0])
                    return

                readTime=cv2.getTickCount()
                interval=(readTime-self.preReadTimeStamp)/cv2.getTickFrequency()
                read_wav = str(ret[4])
                # print('interval',interval,(read_wav==self.preAudioPath))

                if(read_wav!=self.preAudioPath or (read_wav==self.preAudioPath and interval>10)):
                    self.preReadTimeStamp = readTime
                    self.preAudioPath = read_wav
                    self.nativeBookReader.playSound(read_wav)
                    # print ('read it!')

