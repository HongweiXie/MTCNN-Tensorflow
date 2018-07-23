import sys
import numpy as np
sys.path.append('/home/sixd-ailabs/Develop/Human/Hand/Code/build-marker-detect-with-color-pen-c-Desktop_Qt_5_10_0_GCC_64bit-Default')
import ReadBook, cv2

if __name__ == '__main__':

    video_capture = cv2.VideoCapture(1)
    video_capture.set(3, 1280)
    video_capture.set(4, 720)
    bookreader=ReadBook.ReadBook("/home/sixd-ailabs/Develop/Human/Hand/Code/marker-detect-with-color-pen-c/data/2d_recognize/Images","/home/sixd-ailabs/Develop/Human/Hand/Code/marker-detect-with-color-pen-c/data/2d_recognize/book_id_path_tab_config.txt")
    while True:
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, img = video_capture.read()
        image_char = img.astype(np.uint8).tostring()

        bookreader.readPointOnFrame(image_char,img.shape[0], img.shape[1],531,338)
        cv2.imshow("img", img)
        cv2.waitKey(0)