from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def createPattern():

	im = Image.open('test.jpg') # Can be many different formats.
	pix = im.load()
	print(im.size)  # Get the width and hight of the image for iterating over
	x,y = im.size
	areaToCrop = ()
	newIm = im.crop((0,0,30,30))
	newIm.save('template.jpg')
	patternSearch()
	
def patternSearch():

	cvImage = cv2.imread('test.jpg')
	img_gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
	
	
	template = cv2.imread('template.jpg',0)
	w, h = template.shape[::-1]

	result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
	threshold = 0.50
	loc  = np.where(result >= threshold)
	
	for pt in zip(*loc[::-1]):
		print(pt)
		cv2.rectangle(cvImage, pt, (pt[0]+w, pt[1]+h), (0,255,255),2)
		
	cv2.imshow('detected', cvImage)
	cv2.namedWindow('detected',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('detected', 600,600)
	plt.imshow(cvImage)
	plt.show()

if __name__ == "__main__":
	createPattern()