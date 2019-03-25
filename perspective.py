from typing import Optional
import numpy as np
import cv2

from util import WrappedImageProcessor


class PerspectiveTransform:
    """Generates a perspective transformer and 'reverser'"""

    def __init__(self, src: Optional[np.float32] = None, dst: Optional[np.float32] = None):
        if src is None:
            self.src = np.float32([[224, 719],
                                   [619, 440],
                                   [682, 440],
                                   [1100, 719]])
        else:
            self.src = src

        if dst is None:
            self.dst = np.float32([[300, 720],
                                   [300, 0],
                                   [980, 0],
                                   [980, 720]])
        else:
            self.dst = dst

        self.p_transf_matrix = cv2.getPerspectiveTransform(self.src, self.dst)
        self.p_invert_matrix = cv2.getPerspectiveTransform(self.dst, self.src)

    def transform_perspective(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.p_transf_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def restore_perspective(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.p_invert_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    def get_transform_as_processor(self,) -> WrappedImageProcessor:
        """Returns the perspective transform as a ``SimpleImageProcessor``"""
        return WrappedImageProcessor(self.transform_perspective, 'Perspective transform')

    def get_restore_as_processor(self) -> WrappedImageProcessor:
        """Returns the inverse perspective transform as a ``SimpleImageProcessor``"""
        return WrappedImageProcessor(self.restore_perspective, 'Inverse Perspective transform')
