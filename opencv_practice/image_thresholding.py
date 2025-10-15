import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r"c:\Users\ADMIN\Downloads\thanhhoatruemilk.jpg")
img = cv.resize(img,(300,300), interpolation= cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# #Simple thresholding 
# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY) # RET is the threshold value used and thresh1 is the output image
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC) # above 127 are truncated to 127 (cắt nửa trên)
# ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO) # below 127 is set to 0 ( cắt nửa dưới)
# ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

# cv.imshow("Original Image", img)
# cv.imshow("Binary Threshold", thresh1)
# cv.imshow("Binary Inverted Threshold", thresh2)
# cv.imshow("Truncated Threshold", thresh3)
# cv.imshow("To Zero Threshold", thresh4)
# cv.imshow("To Zero Inverted Threshold", thresh5)

#Adaptive Thresholding
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2) # block size is 11 and C is 2  
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2) # 11 is the pixel area
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()













cv.waitKey(0)