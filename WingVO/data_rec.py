import cv2
import numpy as np

from datetime import datetime
# https://www.geeksforgeeks.org/saving-a-video-using-opencv/
now = datetime.now() # current date and time
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap0.isOpened()== False): 
  print("Error stream 0")
if (cap1.isOpened()== False): 
  print("Error stream 1")
frame_width = int(cap0.get(3))
frame_height = int(cap0.get(4))
size = (frame_width, frame_height)
result0 = cv2.VideoWriter(now.strftime("%m_%d_%Y_ %H_%M_%S")+'_c0.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
result1 = cv2.VideoWriter(now.strftime("%m_%d_%Y_ %H_%M_%S")+'_c1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
print(now.strftime("%m/%d/%Y, %H:%M:%S")+'v1.avi')
# Read until video is completed
while(cap0.isOpened() & cap1.isOpened()):
  # Capture frame-by-frame
  ret0, frame0 = cap0.read()
  ret1, frame1 = cap1.read()
  if ret0 == ret1 == True:
 
    # Display the resulting frame
    cv2.imshow('Frame0',frame0)
    cv2.imshow('Frame1',frame1)
    result0.write(frame0)
    result1.write(frame1)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap0.release()
result0.release()
result1.release()
# Closes all the frames
cv2.destroyAllWindows()
