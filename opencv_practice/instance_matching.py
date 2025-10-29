import numpy as np
import cv2 as cv
import os
import random
# import matplotlib.pyplot as plt

def load_templates(directory):
    templates = {}
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg')):
            path = os.path.join(directory, filename)
            template_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            
            if template_img is not None:
                label = os.path.splitext(filename)[0]
                templates[label] = {
                    "image": template_img,
                    "color": (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                }
    return templates

min_match_count = 9

DIR = 'data/template'
templates = load_templates(DIR)
# template_img = cv.imread('data/template/card.jpg', cv.IMREAD_GRAYSCALE)

# orb = cv.ORB_create(nfeatures=2000)
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False) 
# kp1,des1 = orb.detectAndCompute(template_img, None)

akaze = cv.AKAZE_create()
# sift = cv.xfeatures2d.SIFT_create()
# bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=False)
# bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

# kp1, des1 = sift.detectAndCompute(template_img, None)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

detect = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể nhận được khung hình từ camera.")
        break
    key = cv.waitKey(25)
    
    if key == ord('c'):
        roi = cv.selectROI("Select ROI and press ENTER or SPACE", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi

        if w and h:
            cv.destroyAllWindows()
            roi_cropped = frame[y:y+h, x:x+w]
            name = input("Enter label for the template: ")
            img_name = os.path.join(DIR, f"{name}.jpg")
            cv.imwrite(img_name, roi_cropped)

            templates = load_templates(DIR)

    elif key == ord('d'):
        detect = not detect 
        if detect:
            print("detect continuously")
        else:
            print("stop detecting")
    elif key == ord('q'):
        break

    if detect: 
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp2, des2 = akaze.detectAndCompute(gray_frame, None)

        for label, data in templates.items():
            template_img = data["image"]
            template_color = data["color"]

            # cv.imshow(f'Template {label}', template_img)
            kp1, des1 = akaze.detectAndCompute(template_img, None)
            
            matches= bf.knnMatch(des1, des2, k=2)
            matches = sorted(matches, key=lambda x: x[0].distance) # sort to get best matches first
            gud_matches = []
            for m,n in matches:
                if m.distance < 0.80*n.distance:
                    gud_matches.append(m)
                    if len(gud_matches) >= 10:
                        break
            if len(gud_matches) > min_match_count:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in gud_matches ]).reshape(-1,1,2) # src_pts[0] = [x,y]
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in gud_matches ]).reshape(-1,1,2)
                # map points from template image to frame
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0) # RANdom SAmple Consensus , 5 - max distance
                # matchesMask = mask.ravel().tolist()

                if M is not None:
                    h,w = template_img.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    
                    dst = cv.perspectiveTransform(pts,M) 
                    
                    frame = cv.polylines(frame,[np.int32(dst)], True, template_color, 3, cv.LINE_AA)
            else:
                # print(f"[{label}] only {len(gud_matches)} out of {min_match_count}")
                pass

    cv.imshow('object detector', frame)

cap.release()
cv.destroyAllWindows()