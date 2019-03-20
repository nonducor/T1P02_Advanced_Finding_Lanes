from typing import Iterable
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
from moviepy.editor import VideoFileClip
from lane_lines2 import *


class CameraCalibration:
    def __init__(self,):
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def calibrate(self, file_iter: Iterable[str], nx: int = 9, ny: int = 6, debug: bool = True) -> None:
        """Uses all files provided in `file_iter` to generate calibration data.
        `nx` and `ny` are the size of the checkerboard used for calibration"""
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

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
                    cv2.waitKey(500)

        if debug:
            cv2.destroyAllWindows()

        # Finally, calibrate:
        (self.ret,
         self.mtx,
         self.dist,
         self.rvecs,
         self.tvecs) = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)

    def undistort(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))
        return cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

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
    plt.imshow(cal.undistort(cv2.imread('./camera_cal/calibration3.jpg')))
    plt.show()


def run_calibration_and_save():
    cal = CameraCalibration()
    cal.calibrate(glob.glob('./camera_cal/calibration*.jpg'), debug=False)
    cal.save()
    print('Calibration done')


def create_hsl_pipeline(cal):
    def pipeline(img):
        return abs_sobel_thresh(cal.undistort(img))
    return pipeline


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def test_in_all_images(func, folder='test_images'):
    cal = CameraCalibration()
    cal.load()
    for fname in glob.glob(folder + os.sep + '*.jpg'):
        img = cal.undistort(mpimg.imread(fname))
        f_img = np.zeros_like(img)
        f_img[:, :, 0] = 255*func(img)
        comb_img = weighted_img(f_img, img)
        plt.imshow(comb_img)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    direction = (1, 0) if orient == 'x' else (0, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, *direction))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled = np.uint8(255 * mag / np.max(mag))
    binary_output = np.zeros_like(gray)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    absx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    absy = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    img_atan = np.arctan2(absy, absx)
    binary_output = np.zeros_like(gray)
    binary_output[(img_atan >= thresh[0]) & (img_atan <= thresh[1])] = 1
    return binary_output


def test_sobel():
#    f = lambda img: abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
#    f = lambda img: mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))
    f = lambda img: dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    test_in_all_images(f)


def run_full_pipeline(video_file='project_video.mp4', out_folder='./output_videos'):
    cal = CameraCalibration()
    cal.load()

    video = VideoFileClip(video_file).subclip(0, 10)
    pl = create_hsl_pipeline(cal)
    processed_video = video.fl_image(pl)
    processed_video.write_videofile(out_folder + os.sep + video_file, audio=False)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        if not len(sys.argv) > 2:
            print('--test what?')
            exit(1)
        to_test = sys.argv[2]
        if to_test == 'cal':
            test_calibration()
        elif to_test == 'sobel':
            test_sobel()
        else:
            print('Option not understood: %s' % (' '.join(sys.argv[1:])))
            exit(1)
    elif len(sys.argv) > 1:
        if sys.argv[1] == 'cal':
            run_calibration_and_save()
    else:
        run_full_pipeline()
        print('Unknown option')
