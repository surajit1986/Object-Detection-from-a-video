
"""

@author: Surajit
Purpose: a mobile net ssd model to detect object from a video
"""

import numpy as  np
import cv2

#get the video file as stream using cv2
video_stream = cv2.VideoCapture('Sample Video/vd1.mp4')

#create a while loop till the stream is open
while (video_stream.isOpened):
    #get the current frame from video stream
    ret,current_frame = video_stream.read()
    #use the video current frame instead of image
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    print(img_to_detect)
    print(img_height)
    print(img_width)


#releasing the stream
video_stream.release()
cv2.destroyAllWindows()