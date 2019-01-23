import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('prediction_47.png',0)

laplacian = cv2.Laplacian(img,cv2.CV_16S)

plt.imshow(laplacian,cmap = 'gray')
plt.axis('off')
plt.savefig('new.png', bbox_inches='tight', pad_inches=0)	