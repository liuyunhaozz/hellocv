import cv2 as cv
import matplotlib.pyplot as pltfrom 
from PIL import Image
import numpy as np

# img = Image.open('1.jpg')
img = cv.imread('1.jpg')


blur = cv.blur(img,(5,5))
blur0=cv.medianBlur(blur,5)
blur1= cv.GaussianBlur(blur0,(5,5),0)
blur2= cv.bilateralFilter(blur1,9,75,75)

hsv = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)

low_blue = np.array([45, 40, 70])
high_blue = np.array([40, 45, 75])
mask = cv.inRange(hsv, low_blue, high_blue)

res = cv.bitwise_and(img,img, mask= mask)

cv.imshow('hello', mask)
cv.waitKey(0)