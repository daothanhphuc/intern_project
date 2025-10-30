import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

template_img = None 
template_bgr = None
w, h = 0, 0         
lower_threshold = None
upper_threshold = None
detection = 0


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
            
            # code for template matching
            template_img = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)
            w, h = w_roi, h_roi

            # code for hsv colorspace
            hsv_template = cv.cvtColor(template_bgr, cv.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv.split(hsv_template)
            lower_p = 10  
            upper_p = 95 
            h_min_p = np.percentile(h_channel, lower_p)
            h_max_p = np.percentile(h_channel, upper_p)
            s_min_p = np.percentile(s_channel, lower_p)
            s_max_p = np.percentile(s_channel, upper_p)
            v_min_p = np.percentile(v_channel, lower_p)
            v_max_p = np.percentile(v_channel, upper_p)
            lower_threshold = np.array([h_min_p, s_min_p, v_min_p])
            upper_threshold = np.array([h_max_p, s_max_p, v_max_p])
            
            print(f"Đã chụp template mới, kích thước: {w}x{h}")
            # cv.imshow("New Template", template_bgr)
        else:
            print("Hủy chọn template.")
    elif key == ord('q'):
        break
    
    elif key == ord('1'):
        detection = 1
        print("HSV color-based detection")
    elif key == ord('2'):
        detection = 2
        print("Instance matching")    

    if detection == 1:
    # 1.Using hsv colorspace
        if lower_threshold is not None and upper_threshold is not None:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower_threshold, upper_threshold)
            cv.imshow('Mask', mask)
            
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            largest_contour = max(contours, key=cv.contourArea)

            x,y,w,h = cv.boundingRect(largest_contour) # width and height
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            if len(contours) > 0:
                for cnt in contours:
                    rect = cv.minAreaRect(cnt) # ( center (x,y), (width, height), angle)
                    box = cv.boxPoints(rect) # float32
                    box = np.intp(box)
                    cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
    elif detection == 2:
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
    else: 
        pass # do nothing

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