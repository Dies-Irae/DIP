import cv2
import numpy as np
coins = cv2.imread("coins.jpg")
coinsGray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
ret, coinsBin = cv2.threshold(coins, 95, 255, cv2.THRESH_BINARY)

kern = np.ones((5, 5), np.uint8)
coinsEroded = cv2.erode(coinsBin, kernel=kern, iterations=5)
coinsEroded = cv2.cvtColor(coinsEroded, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(coinsEroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
coinsEroded = cv2.cvtColor(coinsEroded, cv2.COLOR_GRAY2BGR)
coinsDrawn = coinsEroded.copy()
coinsDrawn = cv2.drawContours(coinsDrawn, contours, -1, (0,255,0), 3)
res = np.hstack((coins,coinsBin, coinsEroded, coinsDrawn))
cv2.imshow('1', res)
cv2.waitKey(0)
cv2.imwrite('res.jpg', res)
print("The image contains %d coins." % (len(contours)))
