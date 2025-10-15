import cv2 as cv
## Create transparent background of an object based on color detection
img = cv.imread(r"c:\Users\ADMIN\Downloads\ca-betta-mau-xanh-duong.jpg")
img = cv.resize(img,(600,400))
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

#blue and black color range
mask_blue = cv.inRange(hsv,np.array([90,100,100]),np.array([130,255,255])) 
mask_black = cv.inRange(hsv,np.array([0,0,0]),np.array([179,255,100])) # black can only be detected through value channel
# mask = mask_blue + mask_black
mask = cv.bitwise_or(mask_blue,mask_black)

res = cv.bitwise_and(img,img,mask=mask)

b,g,r = cv.split(res)
alpha = mask
rgba = cv.merge([b, g, r, alpha])
# transparent_fish = cv.cvtColor(rgba, cv.COLOR_RGBA2BGRA)

cv.imshow('transparent_fish',rgba)
cv.imshow('img',img)
cv.imshow('mask',mask)
cv.imshow('res',res)

cv.imwrite('transparent_fish.png', rgba)
cv.waitKey(0)
cv.destroyAllWindows()