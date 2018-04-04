import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from PIL import Image


#Load and enhance the image
img = cv.imread('img1.png',0)
equ = cv.equalizeHist(img)


#Find eigen values for ridge filter, save required output as png
hxx, hxy, hyy = hessian_matrix(img, sigma=5)
i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)

plt.imsave('out.png',i1)


#Enhance image again
img2 = cv.imread('out.png',0)
equ2 = cv.equalizeHist(img2)
cv.imshow('veins',equ2)
cv.waitKey(0)
cv.destroyAllWindows()
