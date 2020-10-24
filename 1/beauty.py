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

# r1 = cv2.bilateralFilter(r, 12, 200, 200)
# g1 = cv2.bilateralFilter(g, 12, 200, 200)
# b1 = cv2.bilateralFilter(b, 12, 300, 300)
# filtered = cv2.merge([r, g1, b1])

g2 = cv2.bilateralFilter(g, 5, 100, 100)
b2 = cv2.bilateralFilter(b, 5, 100, 100)
filtered2 = cv2.merge([r, g2, b2])

filtered = cv2.bilateralFilter(img, 10, 100, 75)
filtered = 0.95*filtered + 0.05*filtered2
filtered = filtered.astype(np.uint8)
filtered = adjust_gamma(filtered, 1.4)

plt.subplot(1,2,1), plt.imshow(img)
plt.subplot(1,2,2), plt.imshow(filtered)
plt.pause(100)