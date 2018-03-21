import cv2 as cv
import numpy as np

img = cv.imread('img1.png',0)



th1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)

th = cv.bitwise_not(th1)

#kernel = np.ones((5,5),np.uint8)
#th = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel)
cv.imshow('original',img)
cv.imshow('thresh',th)
cv.waitKey(0)
cv.destroyAllWindows()
