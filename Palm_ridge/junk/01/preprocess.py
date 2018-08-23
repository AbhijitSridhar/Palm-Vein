import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


#Load and enhance the image
img = cv.imread(str(sys.argv[1]),0)
equ = cv.equalizeHist(img)


#Find eigen values for ridge filter, save required output as png
hxx, hxy, hyy = hessian_matrix(equ, sigma=5)
i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
out=str(sys.argv[2])
plt.imsave(out,i1)


#Enhance image again
img2 = cv.imread(out,0)
equ2 = cv.equalizeHist(img2)

th = cv.adaptiveThreshold(equ2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,1)

fin = cv.bitwise_not(th)

cv.imwrite(out,equ2)

cv.imshow('input',img)
cv.imshow('enhanced',equ)
cv.imshow('veins',equ2)
cv.imshow('thresh',fin)
cv.waitKey(0)
cv.destroyAllWindows()

