# 
# # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# 
# 
# cap = cv2.VideoCapture(0)
# 
# while 1:
#     
# 
#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# 
# cap.release()
# cv2.destroyAllWindows()
# 


# import the necessary packages
from __future__ import print_function
from VideoCamera import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2
import numpy as np
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-n", "--num-frames", type=int, default=10,
#     help="# of frames to loop over for FPS test")
# ap.add_argument("-d", "--display", type=int, default=1,
#     help="Whether or not frames should be displayed")
# args = vars(ap.parse_args())

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml') 

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

glasses_cascade = cv2.CascadeClassifier('glasses.xml')

print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)

# loop over some frames...this time using the threaded stream
while True:
    fps = FPS().start()
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
#     frame = imutils.resize(frame, width= 640)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    
    
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(70,245,54),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        
        frame = cv2.putText(frame, "Face Detected", (50,50), font,1, (70,245,54), 2, cv2.LINE_AA)
            
            
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.28, 6)
              
        if len(eyes) == 2:
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(20,250,246),2)
            
            frame = cv2.putText(frame, "Eyes Detected", (50,90), font,1, (20,250,246), 2, cv2.LINE_AA)
        else:    
            
            glasses = glasses_cascade.detectMultiScale(roi_gray, 1.27, 2)
            
            if len(glasses)== 2:
                for (gx,gy,gw,gh) in glasses:
                    cv2.rectangle(roi_color,(gx,gy),(gx+gw,gy+gh),(20,250,246),2)
                    
                frame = cv2.putText(frame, "Glasses and Eyes Detected", (50,90), font,1, (20,250,246), 2, cv2.LINE_AA)    
            else:
                frame = cv2.putText(frame, "No Eyes Detected!", (50,90), font,1, (12,12,245), 2, cv2.LINE_AA)
                
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.6, 20)
        if len(smiles) == 1:
            for (sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,236,64),2)
            frame = cv2.putText(frame, "Smiling!", (50,130), font,1, (255,236,64), 2, cv2.LINE_AA)   
        
        
    else:
        profiles = profile_cascade.detectMultiScale(gray, 1.6, 3)        
        if len(profiles) > 0:
            
            frame = cv2.putText(frame, "Not Looking At Screen!", (50,50), font,1, (12,12,245), 2, cv2.LINE_AA)
            

        
            for (x,y,w,h) in profiles:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(12,12,245),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
            
            
  # check to see if the frame should be displayed to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF 
    # update the FPS counter
    fps.update()
    fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# stop the timer and display FPS information        


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


