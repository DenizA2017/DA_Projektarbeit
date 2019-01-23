import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("prediction_47.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_bl = cv2.bilateralFilter(gray, 9, 75, 75)
ret, thr = cv2.threshold(gray_bl, 200, 255, cv2.THRESH_BINARY_INV)
canvas = np.zeros((1500, 1500), dtype=np.uint8) * 255
canvas[50:50+thr.shape[0], 50:50+thr.shape[1]] = thr
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
dilated = cv2.dilate(canvas, kernel)
_, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
cnt_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
cnt_sorted.pop()
img_drawn = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for i in [range(len(cnt_sorted))[-1]]:
    cv2.drawContours(dilated_rgb, cnt_sorted, i, (0, 255, 0), thickness=5)
    plt.imshow(dilated_rgb)
    plt.show()