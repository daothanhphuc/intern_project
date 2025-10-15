import cv2 as cv
import numpy as np 

background = cv.imread(r"c:\Users\ADMIN\Downloads\130-hinh-nen-may-tinh-4k-7-1024x640.jpg")
transparent_fish = cv.imread(r"Exercise_1\transparent_fish.png", cv.IMREAD_UNCHANGED)
transparent_fish = cv.resize(transparent_fish,(200,150))

h_fg, w_fg, _ = transparent_fish.shape
h_bg, w_bg, _ = background.shape

x_settled = w_bg - w_fg
y_settled = h_bg - h_fg

roi = background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg]

fg_bgr = transparent_fish[:, :,0:3]
alpha = transparent_fish[:, :, 3]

mask_inv = cv.bitwise_not(alpha)

# vị trí của con cá --> đen
bg_masked = cv.bitwise_and(roi, roi, mask=mask_inv)
# lấy vùng hình con cá 
fg_masked = cv.bitwise_and(fg_bgr, fg_bgr, mask=alpha)

composed_roi = cv.add(bg_masked, fg_masked)

background[y_settled:y_settled + h_fg, x_settled:x_settled + w_fg] = composed_roi
cv.imshow("Merged Image", background)
cv.waitKey(0)
cv.destroyAllWindows()