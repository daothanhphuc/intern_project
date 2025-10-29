import cv2 as cv
import numpy as np

# img = cv.imread(r"data/template/facemask.jpg")
# w,h = img.shape[1], img.shape[0]
# print(img.shape) 
# cv.imshow('Template', img)

cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

template_img = None 
w, h = 0, 0         


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display_frame = frame.copy() 

    key = cv.waitKey(1)
    
    if key == ord('c'):
        roi_window_name = "Select ROI and press ENTER or SPACE" 
        roi = cv.selectROI(roi_window_name, frame, fromCenter=False, showCrosshair=True)
        (x, y, w_roi, h_roi) = roi

        if w_roi and h_roi:
            template_bgr = frame[y:y+h_roi, x:x+w_roi]
            cv.destroyWindow(roi_window_name) 
            
            template_img = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)
            
            w, h = w_roi, h_roi
            
            print(f"Đã chụp template mới, kích thước: {w}x{h}")
            cv.imshow("Template Da Chup", template_img)
        else:
            print("Đã hủy chọn template.")
    elif key == ord('q'):
        break
    
    # # 1.Using hsv colorspace
    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # lower_yellow = np.array([20,80,80])
    # upper_yellow = np.array([45,255,255])
    # mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    # cv.imshow('Mask', mask)
    # # largest_contour = max(contours, key=cv.contourArea)
    # contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # # x,y,w,h = cv.boundingRect(largest_contour) # width and height
    # # cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # if len(contours) > 0:
    #     for cnt in contours:
    #         rect = cv.minAreaRect(cnt) # ( center (x,y), (width, height), angle)
    #         box = cv.boxPoints(rect) # float32
    #         box = np.intp(box)
    #         cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

    # 2.Using template matching
    if template_img is not None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h_gray, w_gray = gray.shape

        #  so khớp NẾU frame lớn hơn hoặc bằng template
        if h_gray >= h and w_gray >= w:
            res = cv.matchTemplate(gray, template_img, cv.TM_CCOEFF_NORMED) 
            
            val_min, val_max, loc_min, loc_max = cv.minMaxLoc(res)
            print(f" Max val: {val_max:.2f}, location: {loc_max} ", end="\r")

            threshold = 0.70

            if val_max >= threshold:
                top_left = loc_max
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        else:
            print(f"Bỏ qua frame lỗi: Kích thước frame ({w_gray}x{h_gray}) nhỏ hơn template ({w}x{h})")
            pass 

    cv.imshow('Frame', frame)

cap.release()
cv.destroyAllWindows()



# import cv2
# import os

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Lỗi: Không thể mở camera.")
#     exit()


# img_counter = 0
# save_dir = "data/template"

# while True:
#     ret, frame = cap.read()

#     cv2.imshow('Nhan SPACE de chup, ESC de thoat', frame)

#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:
#         break
#     elif key == 32:
#         img_name = os.path.join(save_dir, f"phone.jpg")
#         cv2.imwrite(img_name, frame)
#         print(f"save image {img_name}")
#         img_counter += 1

# cap.release()
# cv2.destroyAllWindows()