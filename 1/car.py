import cv2
import numpy as np
from matplotlib import pyplot as plt

template = cv2.imread('template.jpg')
templateHsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

car = cv2.imread('car.jpg')
carHsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)

templateHist = cv2.calcHist([templateHsv], [0, 1], None, [180, 10], [0, 179, 0, 256])
cv2.normalize(templateHist, templateHist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([carHsv], [0, 1], templateHist, [0, 179, 0, 256], 1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 4))
dst = cv2.filter2D(dst, -1, disc)

ret, thresh = cv2.threshold(dst, 254, 255, 0)

thresh = cv2.merge((thresh, thresh, thresh))

res = cv2.bitwise_and(car, thresh)
res = np.hstack((car, thresh, res))

cv2.imwrite('res.jpg', res)
cv2.imshow('1', res)
cv2.waitKey(0)
