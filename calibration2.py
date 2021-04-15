import cv2
import glob
import numpy as np

# Implement the number of vertical and horizontal corners
num_vertical = 9
num_horizontal = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((num_horizontal * num_vertical, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_vertical, 0:num_horizontal].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
left_imgpoints = []  # 2d points in image plane.
right_imgpoints = []

framesize = (1280, 720)

left_images = sorted(glob.glob("Stereo_calibration_images/left*"))
right_images = sorted(glob.glob("Stereo_calibration_images/right*"))

for fname in left_images:

    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        left_imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(10)

cv2.destroyAllWindows()

for fname in right_images:

    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        # objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        right_imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(10)

cv2.destroyAllWindows()

# Calibrating cameras

left_ret, left_cam_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(objpoints, left_imgpoints,
                                                                                framesize, None, None)
right_ret, right_cam_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(objpoints, right_imgpoints,
                                                                                     framesize, None, None)

retval, left_cam_matrix, left_dist, right_cam_matrix, right_dist, r, t, e, f = cv2.stereoCalibrate(objpoints,
                                                                                                   left_imgpoints,
                                                                                                   right_imgpoints,
                                                                                                   left_cam_mtx,
                                                                                                   left_dist,
                                                                                                   right_cam_mtx,
                                                                                                   right_dist,
                                                                                                   framesize)

# Undistort left
left_img = cv2.imread(left_images[0])
h,  w = left_img.shape[:2]
# left_new_cam_mtx, left_roi = cv2.getOptimalNewCameraMatrix(left_cam_mtx, left_dist, (w, h), 1, (w, h))
left_dst = cv2.undistort(left_img, left_cam_mtx, left_dist, None)

cv2.imshow('img', left_dst)
cv2.waitKey(0)

# Undistort right
right_img = cv2.imread(right_images[0])
h,  w = right_img.shape[:2]
right_new_cam_mtx, right_roi = cv2.getOptimalNewCameraMatrix(right_cam_mtx, right_dist, (w, h), 1, (w, h))
right_dst = cv2.undistort(right_img, right_cam_mtx, right_dist, None)

cv2.imshow('img', right_dst)
cv2.waitKey(0)

# Rectify
r1, r2, p1, p2, q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(left_cam_mtx,
                                                                      left_dist,
                                                                      right_cam_matrix,
                                                                      right_dist,
                                                                      framesize,
                                                                      r, t)

# Crop the image left
x, y, w, h = valid_pix_roi1
dst = left_dst[y:y+h, x:x+w]
cv2.imwrite('calibrate_result_left.png', dst)

# Crop the image right
x, y, w, h = valid_pix_roi2
dst = right_dst[y:y+h, x:x+w]
cv2.imwrite('calibrate_result_right.png', dst)