import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img = cv2.imread('prediction_47.png',0)
edges = cv2.Canny(img,150,200)

plt.imshow(edges,cmap = 'gray')
plt.axis('off')
plt.savefig('bla.png', bbox_inches='tight', pad_inches=0)	