import cv2
import numpy as np

tape = cv2.imread('tape.jpeg')
tapeGray = cv2.cvtColor(tape, cv2.COLOR_BGR2GRAY)
ret, tapeBin = cv2.threshold(tapeGray, 180, 255, cv2.THRESH_BINARY)

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

tapeEroded = cv2.erode(tapeBin, kernel=kern, iterations=2)
tapeDilated = cv2.dilate(tapeEroded, kernel=kern2, iterations=9)

contours, hierarchy = cv2.findContours(tapeDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

longestContour = contours[0]
for contour in contours:
    if len(longestContour) < len(contour):
        longestContour = contour

tapeDrawn = tape.copy()
tapeDrawn = cv2.drawContours(tapeDrawn, longestContour, -1, (0,255,0), 3)
tapeEroded = cv2.cvtColor(tapeEroded, cv2.COLOR_GRAY2BGR)

res = np.hstack((tapeDrawn,  tapeEroded))

cv2.imshow('1', res)
cv2.waitKey(0)