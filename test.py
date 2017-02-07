import numpy as np
import cv2

img = np.zeros([200,200])

for i in range(0, 200):
    for j in range(np.uint16(i/2), 200-np.uint16(i/2)):
        img[i,j] += 255

img = np.uint8(img)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
