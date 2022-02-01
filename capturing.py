import numpy as np
import cv2
import sys as s

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture(0)
i = 0

while(True):
    _, img = cap.read()
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        img2 = cv2.drawChessboardCorners(img2, (8, 6), corners2, ret)
    cv2.imshow('img', img2)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite('images/chess' + str(i) + '.jpg', img)
        print("Collected image " + str(i))
        i += 1
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        s.exit()
