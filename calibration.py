import glob
import os
import pickle
from typing import Optional, Tuple, Iterable

import cv2
import numpy as np
import matplotlib.pyplot as plt

from util import WrappedImageProcessor


class CameraCalibration:
    def __init__(self,):
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.im_size: Optional[Tuple[int, int]] = None
        self.newcameramtx: Optional[np.ndarray] = None

    def calibrate(self, file_iter: Iterable[str], nx: int = 9, ny: int = 6, debug: bool = True,
                  out_folder: str = 'output_images') -> None:
        """Uses all files provided in `file_iter` to generate calibration data.
        `nx` and `ny` are the size of the checkerboard used for calibration"""
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        im_size = ()
        for fname in file_iter:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im_size = gray.shape

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners if debug is on
                if debug:
                    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    cv2.imshow('img', img)
                    _, out_file_name = os.path.split(fname)
                    out_full_name = os.path.join(out_folder, out_file_name)
                    cv2.imwrite(out_full_name, img)
                    cv2.waitKey(500)

        if debug:
            cv2.destroyAllWindows()

        # Finally, calibrate:
        (self.ret,
         self.mtx,
         self.dist,
         self.rvecs,
         self.tvecs) = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)

    def undistort(self, img) -> np.ndarray:
        cur_im_size = img.shape[:2]

        if self.im_size is None or self.newcameramtx is None or cur_im_size != self.im_size:
            # Regen the camera matrix
            self.im_size = cur_im_size
            h, w = cur_im_size
            self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))

        return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

    def get_undistort_as_processor(self,) -> WrappedImageProcessor:
        """Returns a ``SimpleImageProcessor`` that applies the `undirstort` operation"""
        return WrappedImageProcessor(self.undistort, 'Camera undistortion')

    def save(self, fname: str = 'calibration.pk') -> None:
        with open(fname, 'wb') as fh:
            pickle.dump({'ret': self.ret,
                         'mtx': self.mtx,
                         'dist': self.dist,
                         'rvecs': self.rvecs,
                         'tvecs': self.tvecs}, fh)

    def load(self, fname: str = 'calibration.pk') -> None:
        with open(fname, 'rb') as fh:
            data = pickle.load(fh)
            self.ret = data['ret']
            self.mtx = data['mtx']
            self.dist = data['dist']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']


def test_calibration():
    cal = CameraCalibration()
    cal.calibrate(glob.glob('./camera_cal/calibration*.jpg'))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    orig_img = cv2.imread('./camera_cal/calibration3.jpg')
    ax1.imshow(orig_img)
    ax1.set_title('Before calibration')
    ax2.imshow(cal.undistort(cv2.imread('./camera_cal/calibration3.jpg')))
    ax2.set_title('After calibration')
    plt.savefig('output_images/compare_calibration.png')
    plt.show()


def run_calibration_and_save():
    cal = CameraCalibration()
    cal.calibrate(glob.glob('./camera_cal/calibration*.jpg'), debug=False)
    cal.save()
    print('Calibration done')
