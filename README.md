# Khởi tạo môi trường 
pip install opencv-python
# Get repo 
git clone https://github.com/daothanhphuc/intern_project.git
# Task 1: 
## Merge an object to a specific background
### Extract object wih hsv 
#### Run script for extracting the object
- python .\Exercise_1\get_fish_hsv.py
#### Run script for merging the object
- python .\Exercise_1\merge_image_hsv.py 

### Extract object with contours
#### Run script for extracting the object
- python .\Exercise_1\get_fish_contours.py
#### Run script for merging the object
- python .\Exercise_1\merge_image_contour.py 

## Note of difference between merging image using contours and hsv
### In case of using get_fish_contours.py:
- fg_masked = cv.bitwise_and(fish, fish, mask=mask_inv)
- bg_masked = cv.bitwise_and(roi, roi, mask=mask)

### In case of using get_fish_hsv.py:
- fg_masked = cv.bitwise_and(fish, fish, mask=mask)
- bg_masked = cv.bitwise_and(roi, roi, mask=mask_inv)
