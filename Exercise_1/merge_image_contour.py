import cv2 as cv
import numpy as np 

background = cv.imread(r"data\130-hinh-nen-may-tinh-4k-7-1024x640.jpg")
img = cv.imread(r"data\phone.png")
img = cv.resize(img,(600,400))

img2 = cv.imread(r"data\phone.png")
img2 = cv.resize(img2, (600, 400))

cv.imshow('Original Image', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)
gray = cv.GaussianBlur(gray, (5, 5), 0) # reduce noise
ret, thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY_INV)
cv.imshow('thresh', thresh)

ret, mask = cv.threshold(gray, 90, 140, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

largest_contour = max(contours, key=cv.contourArea)

contours2, hierarchy2 = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img2, contours2, -1, (0,255,0), 2)
cv.imshow('All Contours', img2)
# draw largest contour into a new blank image
mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
cv.drawContours(mask1, [largest_contour], 0, (255,255,255), -1)
result = cv.bitwise_and(img, img, mask=mask1)
cv.imshow('Largest Contour', result) 
# result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

# kernel = np.ones((2,2), np.uint8)
# mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
# cv.imshow("Mask after Morphology", mask)



# fg_bgr = transparent_fish[:, :,0:3] 
# alpha = transparent_fish[:, :, 3]
gray2 = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
ret2, result2 = cv.threshold(gray2, 5, 255, cv.THRESH_BINARY)
cv.imshow("Mask", result2)
contours2, hierarchy2 = cv.findContours(result2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
mask_inv = cv.bitwise_not(result2) 
cv.imshow("Inverted Mask", mask_inv)

# !!! --> parameter 'mask' in bg_masked and fg_masked must be swap when using transparent_fish.png
# vị trí của 'con cá đen' --> sử dụng thresh_inverted
h_fg, w_fg, _ = img.shape
h_bg, w_bg, _ = background.shape

x_settled = w_bg - w_fg
y_settled = h_bg - h_fg

roi = background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg]
bg_masked = cv.bitwise_and(roi, roi, mask=mask_inv)
# lấy vùng hình con cá 
fg_masked = cv.bitwise_and(img, img, mask=result2)

composed_roi = cv.add(bg_masked, fg_masked)

background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg] = composed_roi
cv.imshow("Merged Image", background)
cv.waitKey(0)
cv.destroyAllWindows()