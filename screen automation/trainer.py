import numpy as np
import cv2
from mss import mss
from PIL import Image
import win32api, win32con
import os
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    # win32api.mouse_event(win32con.QS_MOUSEMOVE,x,y,0,0)

def click_data(event, x, y, flags, param):
  if (event == cv2.EVENT_LBUTTONUP):
    click(x,y)
    #  print(x,' , ', y)
    #  font = cv2.FONT_HERSHEY_SIMPLEX
    #  blue = sct_img[y,x,0]
    #  green = sct_img[y,x,1]
    #  red = sct_img[y,x,2]
    #  text = str(red) + ',' + str(green) + ',' + str(blue)
    #  font = cv2.FONT_HERSHEY_SIMPLEX
    #  cv2.putText(sct_img,text,(x,y),font,1,(0,0,255),1,cv2.LINE_AA)
    #  cv2.circle(sct_img, (x,y), radius=0, color=(0, 0, 255), thickness=-1)
    #  cv2.imshow('pickedColor',sct_img)
import os
header_mask_list=[]
cur_dir ="D:/"
for x in os.listdir(cur_dir):

    if x.startswith("tp_"):
        if x.endswith(".png"):
            header_mask = cv2.imread(cur_dir+'/'+x,cv2.IMREAD_GRAYSCALE)
            header_mask_list.append(header_mask)
            class_id = x.split('_')[1].split('.')[0]
            print(class_id)
            class_path = cur_dir+'samples_'+class_id+'/'
            if not os.path.exists(class_path):
                os.mkdir(class_path)

boundaries_cyan = 	([140, 130, 0], [146, 133, 3])
bounding_box = {'top': 0, 'left': 0, 'width': 500, 'height': 900}
tp_bt_human =  cv2.imread('D:/button_human.png',cv2.IMREAD_COLOR)
# tp_bt_human = cv2.cvtColor(tp_bt_human,cv2.COLOR_GRAY2RGB)
tp_label_header =  cv2.imread('D:/label_header.png',cv2.IMREAD_COLOR)
# tp_label_header = cv2.cvtColor(tp_label_header,cv2.COLOR_GRAY2RGB)
tp_selection_corner =  cv2.imread('D:/selection_corner.png',cv2.IMREAD_COLOR)
tp_label_rect = cv2.imread('D:/rect.png',cv2.IMREAD_GRAYSCALE)
# tp_selection_corner = cv2.cvtColor(tp_selection_corner,cv2.COLOR_GRAY2RGB)
#tp_bt_human = cv2.cvtColor(tp_bt_human, cv2.COLOR_BGR2GRAY)
sct = mss()
cur_class=-1
class_frame_count=0
class_samples=[]
cropw = 315
croph = 415
tp_black_rect = cv2.imread('D:/black.png',cv2.IMREAD_COLOR)
while True:
    sct_img = np.asarray(sct.grab(bounding_box))
    sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
    cv2.waitKey(30)
    method = eval('cv2.TM_CCOEFF_NORMED')
    # find header label
    res = cv2.matchTemplate(sct_img,tp_label_header,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if(max_val>0.99):
        tl_point = (max_loc[0], max_loc[1])
        br_point = (max_loc[0]+cropw, max_loc[1]+croph)
        radius = 10
        color = (0,255,0)
        thickness = 2
        sct_img_crop = sct_img[max_loc[1]:max_loc[1]+croph, max_loc[0]:max_loc[0]+cropw]
        # sct_img = cv2.rectangle(sct_img,tl_point,br_point,color=(0, 0, 255), thickness=1)
        
        #cv2.imshow("sct_img_crop",sct_img_crop)
        #find selected corner
        lower, upper = boundaries_cyan
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(sct_img_crop, lower, upper)
        
        #find header class
        header_mask =  mask[0:100, 0:200]
        # cv2.imshow('header_mask', header_mask)
        header_class=-1
        class_id_count = 0
        sum_max_val = 0
        for header_template in header_mask_list:
            res = cv2.matchTemplate(header_mask,header_template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            sum_max_val+=max_val
            if(max_val>0.99):
                header_class=class_id_count
                break
            class_id_count+=1
        if(header_class<0):#no class found
            avr_max_val = sum_max_val/class_id_count
            if(avr_max_val>0.7):
                cv2.imwrite('D:/tp_'+str(len(header_mask_list))+'.png', header_mask) 
                header_mask_list.append(header_mask)
                cv2.imshow('new_template', header_mask)
        
        if  (cur_class!=class_id_count):
            cur_class = (class_id_count)
            class_frame_count = 0
        else: 
            class_frame_count+=1
            #reload class_samples
            if(class_frame_count==3):
                print("cur class:",cur_class)
                class_samples=[]
                class_path = cur_dir+'samples_'+str(cur_class)+'/'
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                for x in os.listdir(class_path):
                    if x.startswith("s_"):
                        if x.endswith(".png"):
                            class_sample = cv2.imread(class_path+'/'+x,cv2.IMREAD_COLOR)
                            class_samples.append(class_sample)
                print("remembered samples:",len(class_samples))
            if(class_frame_count>5):
                # find  news selection
                res = cv2.matchTemplate(mask,tp_label_rect,method)
                (y_points, x_points) = np.where(res >= 0.85)
                #  check if sample exists in class_samples
                k=0
                for (x, y) in zip(x_points, y_points):
                    # print(res[y_points,x_points],x,y)
                    
                    detected_sample = sct_img_crop[ (y+20):(y+95),(x+15):(x+90)]
                    # res = cv2.matchTemplate(detected_sample,tp_black_rect,method)
                    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    # if (max_val>0.9):
                        # continue
                    # sct_img_crop = cv2.rectangle(sct_img_crop, (x+10, y+15), (x+95, y+100), (255, 255, 255), -1)
                    sample_exist = False
                    for sample_img in class_samples:
                        res = cv2.matchTemplate(detected_sample,sample_img,method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        
                        if(max_val>0.7):
                            sample_exist = True
                            break
                    if not sample_exist:
                        cv2.imwrite(class_path+"s_"+str(len(class_samples))+'.png',detected_sample)
                        class_samples.append(detected_sample)
                        print("add sample to class:",cur_class)
                for (x, y) in zip(x_points, y_points):
                    sct_img_crop = cv2.rectangle(sct_img_crop, (x, y), (x+100, y+110), (255, 255, 255), -1)
                #find remembered sample
                for sample_img in class_samples:
                    res = cv2.matchTemplate(sct_img_crop,sample_img,method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if (max_val>0.9):
                        (x, y) = max_loc
                        sct_img_crop = cv2.rectangle(sct_img_crop, (x+10, y+15), (x+95, y+100), (0, 0, 255), 1)
                        cv2.imshow('sample', sample_img)

                    
                    
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # if(max_val>0.75):
            #     center_coordinates = (max_loc[0]+30, max_loc[1]+30)
            #     radius = 5
            #     color = (0,0,255)
            #     thickness = 1
            #     sct_img = cv2.circle(sct_img, center_coordinates, radius, color, thickness)
            # else :
            #     print(max_val)
            # if (cv2.waitKey(1) & 0xFF) == ord('q'):
                #cv2.destroyAllWindows()
                # click(max_loc[0]+10,max_loc[1]+10) #break
        cv2.waitKey(30)
        cv2.imshow('screen', sct_img_crop)

    else:
        cv2.waitKey(30)
        cv2.imshow('screen', sct_img)
    # cv2.setMouseCallback('screen',click_data)


