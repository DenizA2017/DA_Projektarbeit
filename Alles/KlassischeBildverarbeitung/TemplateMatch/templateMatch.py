import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

windowList = os.listdir("windows/")
tileList = os.listdir("tiles/")
templateList = os.listdir("templateTiles/")

globalCounterWindows = 0
globalCounterTiles = 0

for windowImage in windowList:
	if not windowImage.endswith(".DS_Store"):
		img = cv2.imread('./windows/'+str(windowImage)+'')

		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		counterMatches = 0
		for templateImage in templateList:
			if not templateImage.endswith(".DS_Store"):
				template = cv2.imread('./templateTiles/'+str(templateImage)+'',0)

				w, h = template.shape[::-1]

				result = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)

				threshold = 0.70

				loc  = np.where(result >= threshold)

				

				for pt in zip(*loc[::-1]):
					counterMatches += 1
					cv2.circle(img, pt, 20, (255,0,0),2)
	print("Counter Mtach for :"+str(windowImage)+"Count :"+str(counterMatches))
	if counterMatches >10:
		globalCounterWindows += 1

	
print("Counter for Windows: "+str(globalCounterWindows)+" of "+str(len(windowList)) +" Images")
	
	
for tileImage in tileList:
	if not tileImage.endswith(".DS_Store"):
		img = cv2.imread('./tiles/'+str(tileImage)+'')

		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		counterMatches = 0
		for templateImage in templateList:
			if not templateImage.endswith(".DS_Store"):
				template = cv2.imread('./templateTiles/'+str(templateImage)+'',0)
				
				w, h = template.shape[::-1]

				result = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)

				threshold = 0.70

				loc  = np.where(result >= threshold)

				

				for pt in zip(*loc[::-1]):
					counterMatches += 1
					cv2.circle(img, pt, 20, (255,0,0),2)
	print("Counter Mtach for :"+str(tileImage)+" Count: "+str(counterMatches))
	if counterMatches >10:
		globalCounterTiles += 1
		
print("Counter for Tiles: " +str(globalCounterTiles)+" of "+str(len(tileList)) +" Images")



#print("Counter: "+str(counter))	
#cv2.imshow('detected', img)
#cv2.namedWindow('detected',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('detected', 600,600)
#plt.imshow(img)
#plt.show()