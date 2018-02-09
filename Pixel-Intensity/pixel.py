import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bw.jpg',1)

px = img[200,150]
print(px)

imgplot = plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
