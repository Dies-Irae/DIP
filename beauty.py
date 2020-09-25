import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


img = cv2.imread('beauty.jpg')
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

g = cv2.bilateralFilter(g, 5, 200, 200)
b = cv2.bilateralFilter(b, 5, 300, 300)
filtered = cv2.merge([r, g, b])

filtered = cv2.bilateralFilter(img, 10, 100, 75)
filtered = 0.8*filtered + 0.2*img
filtered = filtered.astype(np.uint8)
filtered = adjust_gamma(filtered, 1.4)

plt.subplot(1,2,1), plt.imshow(img)
plt.subplot(1,2,2), plt.imshow(filtered)
plt.pause(100)