import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('data/template/card.jpg')
cv.imshow('image',img)
cv.waitKey(0)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,50,0.01,100)
corners = np.intp(corners) # intp to convert to integer type array
for i in corners:
    x,y = i.ravel() #flatten
    cv.circle(img,(x,y),3,255,-1) #draw circle at corner positions with radius 3, color 255(white) and filled(-1)
plt.imshow(img),plt.show()
