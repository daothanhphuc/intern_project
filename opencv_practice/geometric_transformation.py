import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Resize 
img = cv.imread(r"c:\Users\ADMIN\Downloads\thanhhoatruemilk.jpg")
# INTER_AREA --> shrinking, (INTER_LINEAR, INTER_CUBIC) --> zooming and INTER_NEAREST --> pixel replication
img = cv.resize(img,(300,300), interpolation= cv.INTER_AREA) 
img2 = cv.resize(img,(300,300), interpolation= cv.INTER_NEAREST)
cv.imshow("Using INTER_AREA", img)
cv.imshow("Using INTER_NEAREST", img2)

# Translation --> shifting the image along x and y axis
# x' = m₁₁ * x + m₁₂ * y + m₁₃
# y' = m₂₁ * x + m₂₂ * y + m₂₃
rows,cols = img.shape[:2] # :2 not channels
M = np.float32([[1,0,150],[0,1,50]]) # shift right 150 and down 50
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow("Translated Image", dst)

# Rotation
M = cv.getRotationMatrix2D(((cols-1)*3.0/4.0,(rows-1)*3.0/4.0),90,1) # center, angle, scale
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow("Rotated Image", dst)

# Affine Transformation --> 3 points
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[200,200],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
cv.waitKey(0)


# Perspective Transformation --> 4 points --> using M = cv.getPerspectiveTransform and dst = cv.warpPerspective