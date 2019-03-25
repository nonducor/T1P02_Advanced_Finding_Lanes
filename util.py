from typing import List, Callable, Optional, Dict, Iterable

import cv2
import numpy as np


class ImageProcessor:
    """Abstract class for image processors.

    An image processor provides a function that takes an image and returns a transformed version of that image (the
    transformation is implementation dependent).

    It allows, by a subclass, to chain transformations.
    """
    name: str = 'Image Processor'
    checkpoint_name: Optional[str] = None   # Set to not None to be used in the future (by other pipeline steps)

    def process(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def add(self, proc: 'ImageProcessor') -> 'CompositeImageProcessor':
        raise NotImplementedError

    def has_merge(self) -> Optional[str]:
        """Returns True only if the processor has the capability of receiving 2 images (the current on the pipeline
        and a checkpointed one. Only ``SimpleImageProcessors`` should implement it
        """
        return None


class CompositeImageProcessor(ImageProcessor):
    """An ImageProcessor that executes a series of processors in sequence."""
    def __init__(self, name: str = 'Composite'):
        self.name: str = name
        self.chain: List[ImageProcessor] = []

    def process(self, img: np.ndarray) -> np.ndarray:
        result = img
        saved_imgs: Dict[str, np.ndarray] = {}
        for p in self.chain:
            if isinstance(p, MergingImageProcessor) and p.has_merge() is not None:
                result = p.merge(result, saved_imgs[p.source_name])
            else:
                result = p.process(result)
                if p.checkpoint_name is not None:
                    saved_imgs[p.checkpoint_name] = result.copy()
        return result

    def add(self, proc: ImageProcessor):
        """Add the ImageProcessor as the last element in the processing chain."""
        self.chain.append(proc)
        return self


class SimpleImageProcessor(ImageProcessor):
    """ImageProcessor implementing a simple transformation. Default implementation is Unitary"""
    def __init__(self, name: Optional[str] = None):
        self.name = 'Unitary processor' if name is None else name

    def process(self, img: np.ndarray) -> np.ndarray:
        return img

    def add(self, proc: ImageProcessor) -> CompositeImageProcessor:
        return CompositeImageProcessor().add(self).add(proc)


class MergingImageProcessor(SimpleImageProcessor):
    """An image processor able to consume two parallel images at the same time."""
    source_name: Optional[str] = None  # Name of the second input used by merge

    def connect_merge(self, source_name: Optional[str]):
        """Set the name of a 'variable' in a pipeline to be used by this '2-port' processor."""
        self.source_name = source_name

    def has_merge(self) -> Optional[str]:
        """Returns True only if the processor has the capability of receiving 2 images (the current on the pipeline
        and a checkpointed one. Only ``SimpleImageProcessors`` should implement it
        """
        return self.source_name is not None

    def merge(self, img: np.ndarray, checkpointed_img: np.ndarray) -> np.ndarray:
        """This function will be called by only on classes that have merge capability.
        A checkpointed image will be provided."""
        raise NotImplementedError


class WrappedImageProcessor(SimpleImageProcessor):
    """Utility class that allows a function to be wrapped as a image processor"""
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], description: str = ''):
        SimpleImageProcessor.__init__(self, 'Wrapped function: %s' % description)
        self.func = func

    def process(self, img: np.ndarray) -> np.ndarray:
        return self.func(img)


class CombineImages(MergingImageProcessor):
    """Combine two images.

    The result image is computed as follows:

    img * α + checkpointed_img * β + γ
    NOTE: Both images MUST have the same shape.

    If checkpointed_img is not defined, returns img.
    """
    def __init__(self, α: float = 0.5, β: float = 0.99, γ: float = 0., source_name: Optional[str] = None):
        MergingImageProcessor.__init__(self, 'Combine images')
        self.α = α
        self.β = β
        self.γ = γ
        self.source_name = source_name

    def process(self, img: np.ndarray) -> np.ndarray:
        return img

    def merge(self, img: np.ndarray, checkpointed_img: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(img, self.α, checkpointed_img, self.β, self.γ)


class To3Channel(SimpleImageProcessor):
    """Utility class that gets an image with a single channel and converts it back to tri-channel"""
    def __init__(self, channel: int = 0, normalizer: int = 255):
        SimpleImageProcessor.__init__(self, 'To3Channel in channel %d with norm %d' % (channel, normalizer))
        self.channel = channel
        self.normalizer = normalizer

    def process(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            out_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            out_img[:, :, self.channel] = self.normalizer * img
            return out_img
        else:
            return img


class DrawPoly(SimpleImageProcessor):
    """Draws a polygon in an image. The polygon is fixed and is drawn exactly the same in all images"""
    def __init__(self, points: np.ndarray, color: Iterable[int] = (28, 115, 255), thickness: int = 2):
        self.points = np.int32(points)
        self.color = color
        self.thickness = thickness

    def process(self, img: np.ndarray) -> np.ndarray:
        img_out = img.copy()
        cv2.polylines(img_out, [self.points], True, self.color, self.thickness)
        return img_out
