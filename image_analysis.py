from typing import Tuple

import cv2
import numpy as np
from math import degrees

from util import SimpleImageProcessor


class SobelThreshold(SimpleImageProcessor):
    """Use sobel in one orientation (x or y) with a threshold"""
    def __init__(self, orient: str = 'x', kernel: int = 3, thresh: Tuple[int, int] = (0, 255)):
        SimpleImageProcessor.__init__(self)
        self.direction = (1, 0) if orient == 'x' else (0, 1)
        self.kernel = kernel
        self.thresh = thresh
        self.name = 'Sobel Threshold. dir: %s, kernel: %d, t_min: %d, t_max: %d' % (orient, kernel,
                                                                                    thresh[0], thresh[1], )

    def process(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, *self.direction, ksize=self.kernel))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= self.thresh[0]) & (scaled_sobel <= self.thresh[1])] = 1
        return np.uint8(binary_output)


class MagnitudeThreshold(SimpleImageProcessor):
    """Use absolute sobel kernel value with a threshold"""
    def __init__(self, kernel: int = 3, thresh: Tuple[int, int] = (0, 255)):
        SimpleImageProcessor.__init__(self)
        self.kernel = kernel
        self.thresh = thresh
        self.name = 'Magnitude Sobel Threshold. kernel: %d, t_min: %d, t_max: %d' % (kernel, thresh[0], thresh[1])

    def process(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled = np.uint8(255 * mag / np.max(mag))
        binary_output = np.zeros_like(img)
        binary_output[(scaled >= self.thresh[0]) & (scaled <= self.thresh[1])] = 1
        return np.uint8(binary_output)


class DirectionalThreshold(SimpleImageProcessor):
    """Use a threshold dependent on the derivative direction"""
    def __init__(self, kernel: int = 3, thresh: Tuple[float, float] = (0.0, np.pi / 2)):
        SimpleImageProcessor.__init__(self)
        self.kernel = kernel
        self.thresh = thresh
        self.name = 'Directional Sobel Thresh. kernel: %d, min_angle: %.2f, max_angle: %.2f' % (kernel,
                                                                                                degrees(thresh[0]),
                                                                                                degrees(thresh[1]))

    def process(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        absx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel))
        absy = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel))
        img_atan = np.arctan(absx / absy)
        binary_output = np.zeros_like(img)
        binary_output[(img_atan >= self.thresh[0]) & (img_atan <= self.thresh[1])] = 1
        return np.uint8(binary_output)


class CombinedAnalysis(SimpleImageProcessor):
    """A processor that takes into account both gradient directions and absolute values"""
    def __init__(self, proc_color: SimpleImageProcessor = None, proc_x: SobelThreshold = None, proc_y: SobelThreshold = None,
                 proc_mag: MagnitudeThreshold = None, proc_dir: DirectionalThreshold = None):
        SimpleImageProcessor.__init__(self)
        self.proc_color = SRThreshold() if proc_color is None else proc_color
        self.proc_x = SobelThreshold(orient='x', kernel=7, thresh=(15, 70)) if proc_x is None else proc_x
        self.proc_y = SobelThreshold(orient='y', kernel=7, thresh=(15, 70)) if proc_y is None else proc_y
        self.proc_mag = MagnitudeThreshold(kernel=9, thresh=(15, 70)) if proc_mag is None else proc_mag
        self.proc_dir = DirectionalThreshold(kernel=15, thresh=(np.radians(30), np.radians(60.0))) if proc_dir is None else proc_dir
        self.name = 'Combined Analysis'

    def process(self, img: np.ndarray) -> np.ndarray:
        # Apply each of the thresholding functions
        color = self.proc_color.process(img)
        gradx = self.proc_x.process(img)
        grady = self.proc_y.process(img)
        mag_binary = self.proc_mag.process(img)
        dir_binary = self.proc_dir.process(img)
        combined = np.zeros_like(dir_binary)
        combined[(color == 1) | (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
        return np.uint8(combined)


class HChannelThreshold(SimpleImageProcessor):
    """Uses the H channel in a HLS representation to find lane markings"""
    def __init__(self, thresh: Tuple[float, float] = (15, 30)):
        SimpleImageProcessor.__init__(self)
        self.thresh = thresh
        self.name = 'H chan threshold. t_min: %d, t_max: %d' % (thresh[0], thresh[1],)

    def process(self, img: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_chan = hls[:, :, 0]
        binary = np.zeros_like(h_chan)
        binary[(h_chan > self.thresh[0]) & (h_chan <= self.thresh[1])] = 1
        return np.uint8(binary)


class SChannelThreshold(SimpleImageProcessor):
    """Uses the S channel in a HLS representation to find lane markings"""
    def __init__(self, thresh: Tuple[float, float] = (90, 255)):
        SimpleImageProcessor.__init__(self)
        self.thresh = thresh
        self.name = 'S chan threshold. t_min: %d, t_max: %d' % (thresh[0], thresh[1],)

    def process(self, img: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_chan = hls[:, :, 2]
        binary = np.zeros_like(s_chan)
        binary[(s_chan > self.thresh[0]) & (s_chan <= self.thresh[1])] = 1
        return np.uint8(binary)


class RChannelThreshold(SimpleImageProcessor):
    """Uses the R channel in a RGB representation to find lane markings"""
    def __init__(self, thresh: int = 180):
        SimpleImageProcessor.__init__(self)
        self.thresh = thresh
        self.name = 'R chan threshold. thresh above: %d' % thresh

    def process(self, img: np.ndarray) -> np.ndarray:
        r_chan = img[:, :, 0]
        binary = np.zeros_like(r_chan)
        binary[r_chan > self.thresh] = 1
        return np.uint8(binary)


class SRThreshold(SimpleImageProcessor):
    """A processor that uses both R and S thresholding"""
    def __init__(self, proc_s: SChannelThreshold = None, proc_r: RChannelThreshold = None):
        SimpleImageProcessor.__init__(self)
        self.proc_s = SChannelThreshold() if proc_s is None else proc_s
        self.proc_r = RChannelThreshold() if proc_r is None else proc_r
        self.name = 'Combined S & R Thresholding'

    def process(self, img: np.ndarray) -> np.ndarray:
        s_thresholded = self.proc_s.process(img)
        r_thresholded = self.proc_r.process(img)
        return np.uint8(s_thresholded & r_thresholded)


class SRPlusSobelX(SimpleImageProcessor):
    """S&R analysis OR Gradient in X"""
    def __init__(self, proc_sr: SRThreshold = None, proc_x: SobelThreshold = None,):
        SimpleImageProcessor.__init__(self)
        self.proc_sr = SRThreshold() if proc_sr is None else proc_sr
        self.proc_x = SobelThreshold(orient='x', kernel=7, thresh=(30, 100)) if proc_x is None else proc_x
        self.name = '(S & R) | GradX Combined Analysis'

    def process(self, img: np.ndarray) -> np.ndarray:
        # Apply each of the thresholding functions
        sr = self.proc_sr.process(img)
        gradx = self.proc_x.process(img)
        combined = np.zeros_like(gradx)
        combined[(sr == 1) | (gradx == 1)] = 1
        return np.uint8(combined)
