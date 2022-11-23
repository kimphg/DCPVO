import numpy as np
import cv2
from mss import mss
from PIL import Image
import win32api, win32con
sct = mss()
bounding_box = {'top': 0, 'left': 0, 'width': 500, 'height': 900}
sct_img = np.asarray(sct.grab(bounding_box))

def click_data(event, x, y, flags, param):

  if (event == cv2.EVENT_LBUTTONDOWN):
     print(x,' , ', y)
     font = cv2.FONT_HERSHEY_SIMPLEX
     blue = sct_img[y,x,0]
     green = sct_img[y,x,1]
     red = sct_img[y,x,2]
     text = str(red) + ',' + str(green) + ',' + str(blue)
     font = cv2.FONT_HERSHEY_SIMPLEX
     cv2.putText(sct_img,text,(x,y),font,1,(0,0,255),1,cv2.LINE_AA)
     cv2.circle(sct_img, (x,y), radius=0, color=(0, 0, 255), thickness=-1)
     cv2.imshow('imagename',sct_img)




while True:
    sct_img = np.asarray(sct.grab(bounding_box))
    cv2.imshow('screen', sct_img)
    cv2.setMouseCallback('screen',click_data)
    cv2.waitKey(30)
    
