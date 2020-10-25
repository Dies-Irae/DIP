import cv2
import numpy as np
import math


def distance(x, y): # return squared euclid distance
    return (x[0]-y[0])**2 + (x[1]-y[1])**2


tape = cv2.imread('tape.jpeg')
tapeGray = cv2.cvtColor(tape, cv2.COLOR_BGR2GRAY)
ret, tapeBin = cv2.threshold(tapeGray, 180, 255, cv2.THRESH_BINARY)

kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

tapeEroded = cv2.erode(tapeBin, kernel=kern, iterations=2)
tapeDilated = cv2.dilate(tapeEroded, kernel=kern2, iterations=7)
cv2.imwrite('tapeDilated.jpg', tapeDilated)

contours, hierarchy = cv2.findContours(tapeDilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

longestContour = contours[0]
for contour in contours:
    if len(longestContour) < len(contour):
        longestContour = contour

tapeDrawn = tape.copy()

(x, y), radius = cv2.minEnclosingCircle(longestContour)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(tapeDrawn, center, radius, (0,255,0), 2)
longestContour = np.squeeze(longestContour)
longestContour = longestContour.tolist()

innerCircle = []
for x in longestContour:
    if distance(x, center) < (radius-50)**2:
        innerCircle.append(x)

distanceSum = 0
for x in innerCircle:
    distanceSum += distance(x, center)

# innerCircle = np.reshape(innerCircle, (len(innerCircle), 1, 2))
# tapeDrawn = cv2.drawContours(tapeDrawn, innerCircle, -1, (0,255,0), 3)

estimatedR = math.sqrt(distanceSum/len(innerCircle))
estimatedR = int(estimatedR)
cv2.circle(tapeDrawn, center, estimatedR, (0,255,0), 2)
cv2.imwrite('tapeDrawn.jpg', tapeDrawn)

cv2.waitKey(0)