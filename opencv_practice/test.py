import cv2 as cv
import numpy as np
import os

TEMPLATE_DIR = "data/template"
CONFIDENCE_THRESHOLD = 0.6

def load_templates(directory):
    templates = {}

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg')):
            path = os.path.join(directory, filename)
            template_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            
            if template_img is not None:
                label = os.path.splitext(filename)[0]
                w, h = template_img.shape[::-1]
                templates[label] = (template_img, (w, h))
                print(f"template {label}, {w}x{h}")
    return templates

templates = load_templates(TEMPLATE_DIR)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    for label, (template, (w, h)) in templates.items():
        res = cv.matchTemplate(gray_frame, template, cv.TM_CCOEFF_NORMED) 
        loc = np.where(res >= CONFIDENCE_THRESHOLD) # [(y1, y2, ...),(x1, x2, ...)], 
        
        for pt in zip(*loc[::-1]):  # pt là (x, y)
            if label == "card":
                cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            elif label == "watch":  
                cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

    cv.imshow('All obj', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()