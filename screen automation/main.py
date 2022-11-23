import numpy as np
import cv2
from mss import mss
from PIL import Image
import win32api, win32con
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    # win32api.mouse_event(win32con.QS_MOUSEMOVE,x,y,0,0)


bounding_box = {'top': 0, 'left': 0, 'width': 500, 'height': 900}
tp_bt_human =  cv2.imread('D:/button_human.png',0)
#tp_bt_human = cv2.cvtColor(tp_bt_human, cv2.COLOR_BGR2GRAY)
sct = mss()

while True:
    sct_img = np.asarray(sct.grab(bounding_box))
    sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
    
    method = eval('cv2.TM_CCOEFF_NORMED')
    res = cv2.matchTemplate(sct_img,tp_bt_human,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if(max_val>0.5):
        # cv2.imshow('res', (res))
        # Center coordinates
        center_coordinates = (max_loc[0]+30, max_loc[1]+30)
        # Radius of circle
        radius = 10
        # Blue color in BGR
        color = (0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        sct_img = cv2.circle(sct_img, center_coordinates, radius, color, thickness)
  
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            #cv2.destroyAllWindows()
            click(max_loc[0]+10,max_loc[1]+10) #break
    else:
        cv2.waitKey(30)
    cv2.imshow('screen', sct_img)
    
    