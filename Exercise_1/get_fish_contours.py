import numpy as np
import cv2 as cv

img = cv.imread(r"data\ca-betta-mau-xanh-duong.jpg")
img = cv.resize(img,(600,400))
cv.imshow('Original Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 135, 255, cv.THRESH_BINARY_INV)
cv.imshow('thresh', thresh)

# kernel = np.ones((5,5), np.uint8) # Tạo một kernel 5x5
# closed_thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
# cv.imshow('Closed Thresh', closed_thresh)

# findContours contains image/contours_mode/contours_method 
# Options for mode: RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE --> CCOMP - detect holes in objects
# Options for method: CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

largest_contour = max(contours, key=cv.contourArea)
# print(f'Total contours found: {len(contours)}')

# cv.drawContours(image, contours, contourIdx, color, thickness) --> image is destination image
cv.drawContours(img, [largest_contour], -1, (0,0,255), 0) 

cv.imshow('contours', img)
cv.imwrite('data/fish_contour.png', img)
cv.waitKey(0)
cv.destroyAllWindows()