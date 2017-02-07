import numpy as np
import cv2
import time

# Library to read and process video
cap = cv2.VideoCapture("red_line_vid.mp4")

def processing(img):
    # Ideally line has to be vertical in the image. Transpose if necessary.
    img = cv2.transpose(img)

    # Convert to HSV. First term determines the color. Red range is 0-30(ish).
    # Other values determine saturation and brightness.
    # http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Min and max of threshold.
    minBound = np.array([0, 140, 140])
    maxBound = np.array([20, 255, 255])
    # Creates mask. 1 for pixels within bounds, 0 otherwise.
    mask = cv2.inRange(img_hsv, minBound, maxBound)

    res = np.copy(img)
    # List of coordinates of all points within the threshold (on the line).
    pline = []
    height, width, c = img.shape
    # for i in range(0, img.shape[0]-200,2):
    #     for j in range(0, img.shape[1],3):
    # This for loop has the triangle optimisation set up.
    for i in range(0, height, 1):
        for j in range(np.uint16(i/2), width - np.uint16(i/2), 2):
            if (mask[i,j] > 0):
                # res[i,j] = np.array([0, 255, 0])
                pline.append([i, j])

    pline = np.array(pline)

    # The two matrices for the least squares
    # X has two colums: a column of ones, and a column of the x coordinates of the points within theshold
    X = np.array([np.ones(pline.shape[0]), pline[:,0]]).T
    # y has one colums: the y coordinate of the points within threshold
    y = np.array(pline[:, 1]).reshape(-1, 1)
    coeffs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # Line coefficients
    c = coeffs[0]
    m = coeffs[1]

    # Drawing the line of best fit
    pt1 = (c, 0)
    pt2 = (height*m + c, height)
    cv2.line(res, pt1, pt2, color=[255, 0, 0], thickness=3)

    # Drawing the black line in the center of the screen
    pt1 = (np.int32(width/2), 0)
    pt2 = (np.int32(width/2), height)
    cv2.line(res, pt1, pt2, color=[0,0,0], thickness=2)

    return res

framesSinceStart = 0
startTime = time.time()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (400, 300))

    # Our operations on the frame come here
    out = processing(frame)

    currentTime = time.time()
    fps = np.int8(np.round(framesSinceStart / (currentTime - startTime)))
    cv2.putText(out, "FPS: " + str(fps), (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    framesSinceStart += 1

    # Display the resulting frame
    cv2.imshow('frame',out)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
