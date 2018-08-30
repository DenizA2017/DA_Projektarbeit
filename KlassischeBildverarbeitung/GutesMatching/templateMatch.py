import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('schaden7.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('schaden.jpg',0)
template2 = cv2.imread('template2.jpg',0)

w, h = template.shape[::-1]

result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
result2 = cv2.matchTemplate(img_gray, template2, cv2.TM_CCOEFF_NORMED)

threshold = 0.35

loc  = np.where(result >= threshold)
loc2  = np.where(result2 >= threshold)


for pt in zip(*loc[::-1]):
	print(pt)
	cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,255),2)
	
#for pt in zip(*loc2[::-1]):
	#cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255,255,0),2)
	
	
cv2.imshow('detected', img)
cv2.namedWindow('detected',cv2.WINDOW_NORMAL)
cv2.resizeWindow('detected', 600,600)
plt.imshow(img)
plt.show()