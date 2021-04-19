import cv2
import glob
import os
import numpy as np


def calibrate_cam(calibration_imgs):

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
    imgpoints = []  # 2d points in image plane.

    framesize = (1280, 720)

    for fname in calibration_imgs:

        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(10)

    cv2.destroyAllWindows()

    # Calibration
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)

    return objpoints, imgpoints, ret, cam_mtx, dist, rvecs, tvecs


def undistort(images, camera_side, cam_mtx, dist, valid_pix_roi):

    for i in range(len(images)):

        fname = images[i]
        img = cv2.imread(images[i])
        dst = cv2.undistort(img, cam_mtx, dist, None)
        x, y, w, h = valid_pix_roi
        dst = dst[y:y + h, x:x + w]

        if camera_side == "left":

            fname = fname.split("left/")[1]
            os.chdir("/Users/olemartinsorensen/PycharmProjects/Perception-for-AS-Final-Project/PAS-project/Calibrated_Stereo_With_Occlusions/Left")
            cv2.imwrite(f"{fname}", dst)
            os.chdir("..")
            os.chdir("..")

        else:

            fname = fname.split("right/")[1]
            os.chdir("/Users/olemartinsorensen/PycharmProjects/Perception-for-AS-Final-Project/PAS-project/Calibrated_Stereo_With_Occlusions/Right")
            cv2.imwrite(f"{fname}", dst)
            os.chdir("..")
            os.chdir("..")


def main():

    framesize = (1280, 720)

    left_images = sorted(glob.glob("Stereo_calibration_images/left*"))
    right_images = sorted(glob.glob("Stereo_calibration_images/right*"))

    objpoints, left_imgpoints, left_ret, left_cam_mtx, left_dist, left_rvecs, left_tvecs = calibrate_cam(left_images)
    _, right_imgpoints, right_ret, right_cam_mtx, right_dist, right_rvecs, right_tvecs = calibrate_cam(right_images)

    # Stereo calibration
    retval, left_cam_matrix, left_dist, right_cam_matrix, right_dist, r, t, e, f = cv2.stereoCalibrate(objpoints,
                        left_imgpoints, right_imgpoints, left_cam_mtx, left_dist, right_cam_mtx, right_dist, framesize)

    # Rectify
    r1, r2, p1, p2, q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(left_cam_mtx,
                        left_dist, right_cam_matrix, right_dist, framesize, r, t)

    left_img = sorted(glob.glob("Stereo_conveyor_with_occlusions/left/*"))
    right_img = sorted(glob.glob("Stereo_conveyor_with_occlusions/right/*"))

    # Undistort
    undistort(left_img, "left", left_cam_matrix, left_dist, valid_pix_roi1)
    undistort(right_img, "right", right_cam_matrix, right_dist, valid_pix_roi2)


if __name__ == '__main__':

    main()
