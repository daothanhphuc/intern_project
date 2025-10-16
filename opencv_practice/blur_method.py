# 1. Average Blurring
# 2. Gaussian Blurring  -> weighted average, central value has more weight
# 3. Median Blurring    -> replace central value with median of surrounding pixels
# 4. Bilateral Blurring -> best for edge preserving

import cv2 as cv
import numpy as np 

img = cv.imread('data/ca-betta-mau-xanh-duong.jpg')
img = cv.resize(img, (600,400))
cv.imshow('Original', img)

def padding(img, pad_size):
    h, w = img.shape[:2]
    pad_img = np.zeros((h + 2*pad_size, w + 2*pad_size, 3), dtype=img.dtype) 
    pad_img[pad_size:pad_size+h, pad_size:pad_size+w] = img
    return pad_img

#implement from scratch
def average_blur(image, kernel_size): 
    kernel = np.ones((kernel_size,kernel_size,3), dtype=np.float32)
    kernel = kernel/(kernel_size*kernel_size)
    height, width = image.shape[:2]
    # print(height, width)
    pad_img = padding(image, kernel_size//2)
    out = np.zeros_like(pad_img)
    for i in range(height):
        for j in range(width):
            roi = pad_img[i:i+kernel_size, j:j+kernel_size]
            out[i,j] = np.sum(roi*kernel, axis=(0,1) )
    height,width = out.shape[:2]
    final = out[0:height - 2*(kernel_size//2 ), 0:width - 2*(kernel_size//2 )]
    return final

average = cv.blur(img, (5,5))
cv.imshow('Average Blurring', average)
print(average.shape)

blur_from_scratch = average_blur(img, kernel_size=5)
cv.imshow('Average Blurring from scratch', blur_from_scratch)
print(blur_from_scratch.shape)

cv.waitKey(0)
cv.destroyAllWindows()
    