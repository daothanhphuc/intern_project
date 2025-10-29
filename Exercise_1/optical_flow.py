#optical flow with direction
# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255
# while(1):
#     ret, frame2 = cap.read()
#     next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
#     cv.imshow('frame2',bgr)
#     k = cv.waitKey(30) & 0xff # 30 means
#     if k == 27:
#         break
#     # elif k == ord('s'):
#     #     cv.imwrite('opticalfb.png',frame2)
#     #     cv.imwrite('opticalhsv.png',bgr)
#     prvs = next

# capture image from cam
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
if not ret:
    print("Lỗi: Không thể đọc frame đầu tiên.")
    cap.release()
    exit()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# 1.0 Rất nhạy, phát hiện cả nhiễu.
# 5.0 Chỉ phát hiện chuyển động nhanh.
MOVEMENT_THRESHOLD = 3.0

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print("end")
        break

    next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

    ret_thresh, move_mask = cv.threshold(mag, MOVEMENT_THRESHOLD, 255, cv.THRESH_BINARY)

    move_mask_uint8 = move_mask.astype(np.uint8)

    cv.imshow('Movement', move_mask_uint8)
    cv.imshow('Original', frame2) 

    k = cv.waitKey(30) 
    if k == 27:
        break

    prvs = next_frame

cap.release()
cv.destroyAllWindows()
