import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("red_line.jpg")

vid = cv.VideoCapture.open("red_line_vid.jpg")

img = cv.resize(img, (806,604))

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

minBound = np.array([0, 140, 140])
maxBound = np.array([20, 255, 255])

mask = cv.inRange(img_hsv, minBound, maxBound)

res = np.copy(img)
pline = []
for i in range(0, img.shape[0],4):
    for j in range(0, img.shape[1],4):
        if (mask[i,j] > 0):
            res[i,j] = np.array([0, 255, 0])
            pline.append([i, j])

pline = np.array(pline)

X = np.array([np.ones(pline.shape[0]), pline[:,0]]).T
y = np.array(pline[:, 1]).reshape(-1, 1)
coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

c = coeffs[0]
m = coeffs[1]

print("c: " + str(c))
print("m: " + str(m))

height = res.shape[0]
width = res.shape[1]
pt1 = (c, 0)
pt2 = (height*m + c, height)
cv.circle(res, pt1, 3, color=[0,0,255])
cv.line(res, pt1, pt2, color=[255, 0, 0], thickness=3)
pt1 = (np.int32(width/2), 0)
pt2 = (np.int32(width/2), height)
cv.line(res, pt1, pt2, color=[0,0,0], thickness=2)

cv.imshow("Original", img)
cv.imshow("Result", res)
cv.waitKey(0)
cv.destroyAllWindows()
