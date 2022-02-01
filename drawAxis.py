import numpy as np
import cv2
import glob
import math
import sys as s

# Load previously saved data
with np.load('calibration/calibdata.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        x = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = math.degrees(0)
    return np.array([x, y, z])

cap = cv2.VideoCapture(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

while(True):
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        if ret:
            rotMatrix, _ = cv2.Rodrigues(rvecs)
            angles = rotationMatrixToEulerAngles(rotMatrix)
            print("Rotational Angles: " + str(angles))

            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            if np.all(imgpts < 1000):
                hls = draw(img, corners2, imgpts)
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        s.exit()
