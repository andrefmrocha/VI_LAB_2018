import numpy
import cv2

img = cv2.imread('0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image',gray)
cv2.waitKey(0)
