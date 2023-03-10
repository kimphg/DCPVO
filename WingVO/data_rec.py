import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
# Check if camera opened successfully
if (cap0.isOpened()== False): 
  print("Error stream 0")
if (cap1.isOpened()== False): 
  print("Error stream 1")
frame_width = int(cap0.get(3))
frame_height = int(cap0.get(4))
size = (frame_width, frame_height)
result0 = cv2.VideoWriter(strftime()+'v0.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
result1 = cv2.VideoWriter(strftime()+'v1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# Read until video is completed
while(cap0.isOpened() & cap1.isOpened()):
  # Capture frame-by-frame
  ret0, frame0 = cap0.read()
  ret1, frame1 = cap1.read()
  if ret0 == ret1 == True:
 
    # Display the resulting frame
    cv2.imshow('Frame0',frame0)
    cv2.imshow('Frame1',frame1)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
