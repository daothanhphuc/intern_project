import cv2 as cv
import numpy as np 

#HSV --> Hue - Saturation - Value with range of Hue is [0,179], Saturation is [0,255] and Value is [0,255]
# --> require normalization of ranges
# Hue is the color type                     --> 0 is red, 60 is green, 120 is blue and magenta is 150
# Saturation is the intensity of the color  --> 0 is black and 255 is the actual color
# Value is the brightness of the color      --> 0 is completely dark and 255 is completely bright

# cap = cv.VideoCapture(0)
# while(1):
#     _,frame = cap.read()
#     hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

#     # get yellow and green color range
#     lower_yellow = np.array([20,50,50])
#     upper_yellow = np.array([40,255,255])
#     # create a mask for yellow color
#     lower_green = np.array([40,100,100])
#     upper_green = np.array([70,255,255])


#     mask = cv.inRange(hsv,lower_yellow,upper_green)
#     # apply bitwise and to get only yellow 
#     res = cv.bitwise_and(frame,frame,mask=mask)
#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     # press esc to exit
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()

# # get hsv value of a specific color 
# # green = np.uint8([[[0,255,0 ]]])
# # blue = np.uint8([[[255,0,0 ]]])
# #red = np.uint8([[[0,0,255 ]]])
# yellow = np.uint8([[[0,255,255 ]]])
# hsv = cv.cvtColor(yellow,cv.COLOR_BGR2HSV)
# print( hsv )
