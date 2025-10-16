import cv2 as cv
import numpy as np 

background = cv.imread(r"data\130-hinh-nen-may-tinh-4k-7-1024x640.jpg")
fish = cv.imread(r"data\transparent_fish.png")
# fish = cv.imread(r"data\fish_contour.png")
fish = cv.resize(fish,(200,150))
cv.imshow("Fish", fish)

gray_fish = cv.cvtColor(fish, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(gray_fish, 5, 255, cv.THRESH_BINARY)
cv.imshow("Mask", mask)

h_fg, w_fg, _ = fish.shape
h_bg, w_bg, _ = background.shape

x_settled = w_bg - w_fg
y_settled = h_bg - h_fg

roi = background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg]

# fg_bgr = transparent_fish[:, :,0:3]
# alpha = transparent_fish[:, :, 3]

mask_inv = cv.bitwise_not(mask) 
cv.imshow("Inverted Mask", mask_inv)

# !!! --> parameter 'mask' in bg_masked and fg_masked must be swap when using transparent_fish.png
# vị trí của 'con cá đen' --> sử dụng thresh_inverted
bg_masked = cv.bitwise_and(roi, roi, mask=mask_inv)
# lấy vùng hình con cá 
fg_masked = cv.bitwise_and(fish, fish, mask=mask)

composed_roi = cv.add(bg_masked, fg_masked)

background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg] = composed_roi
cv.imshow("Merged Image", background)
cv.waitKey(0)
cv.destroyAllWindows()