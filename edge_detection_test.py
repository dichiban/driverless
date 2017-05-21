import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Ch2_001/center/1479425441182877835.jpg',0)

edges = cv2.Canny(img,100,200)

#Simply plot the original image and the edge detection (grey scaled)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()