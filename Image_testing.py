import numpy as np
import cv2

i = 0
img = cv2.imread('0.png')
kernel = np.ones((7, 7), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (5, 5))
ret, thresh = cv2.threshold(gray, 200, 255, 1)
erosion = cv2.erode(thresh, kernel, iterations=3)
dilation = cv2.dilate(erosion, kernel, iterations=1)
# dilation = cv2.dilate(gray, kernel, iterations =2)
# while(i < 10):
#      erosion = cv2.erode(dilation, kernel, iterations=3)
#      dilation = cv2.dilate(erosion, kernel, iterations=4)
#      i+=1
# cv2.imshow('image',gray)
# cv2.waitKey(0)
# erosion = cv2.erode(dilation,kernel,iterations = 4)

image, contour, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contour))
cv2.drawContours(image, contour, -1, (0, 255,0), 3)
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', image)
cv2.waitKey(0)
