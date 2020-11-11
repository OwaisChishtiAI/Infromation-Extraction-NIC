import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
#initializing Background subtractor
fgbg = cv2.createBackgroundSubtractorKNN()
  

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Making Rectangle
    cv2.rectangle(frame, (100, 110), (510, 390), (0, 20, 200), 10)
   #Applying Mask for background detection
    fgmask = fgbg.apply(frame[110:390 , 100:510])
    #Applying Canny Edge detector
    edges = cv2.Canny(frame[110:390 , 100:510],100,150,apertureSize = 3)
    # Display the resulting frame
    cv2.imshow('Canny Edge Detector',edges)
    cv2.imshow('Live Stream',frame)
    cv2.imshow('Backgrounf Subtraction', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()