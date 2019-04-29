from typing import Iterable, Union, List, Optional, Sequence, Tuple
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2
from math import radians
from moviepy.editor import VideoFileClip

from calibration import CameraCalibration, test_calibration, run_calibration_and_save
from perspective import PerspectiveTransform
from util import WrappedImageProcessor, ImageProcessor, CombineImages, MergingImageProcessor, To3Channel, \
    SimpleImageProcessor, DrawPoly
from image_analysis import SChannelThreshold, RChannelThreshold, HChannelThreshold, SRThreshold, SobelThreshold, \
    MagnitudeThreshold, CombinedAnalysis, DirectionalThreshold


class LaneFindingConfig:
    """Data holder class for some meta params used by the lane find algorithm"""

    def __init__(self, n_windows: int = 9,
                 margin: int = 100,
                 minpixels: int = 50,
                 hist_smoothing: int = 30,
                 min_lane_separation_px: int = 400,
                 max_unstable_frames: int = 4,
                 lane_width: float = 3.7,
                 distance_to_horizon: float = 40.0,  # Visually measured using lane markings (12 m btw marking start)
                 max_parallel_diff: float = 1.0,     # Max error from bottom to top
                 image_size: Optional[Tuple[int, int]] = None,):
        self.n_windows = n_windows
        self.margin = margin
        self.minpixels = minpixels   # Min num pixels to update sliding window position
        self.hist_smoothing = hist_smoothing
        self.min_lane_separation_px = min_lane_separation_px
        self.max_unstable_frames = max_unstable_frames
        self.lane_width = lane_width
        self.distance_to_horizon = distance_to_horizon
        self.max_parallel_diff = max_parallel_diff   # In meters
        self.mx: float = 1.0  # Conversion factor: m/px for x dimension (to be updated later)
        self.my: float = distance_to_horizon / 720  # Conversion factor: m/px for y dimension

        # Parameters below are not strictly configuration, but cached values
        self.image_size = image_size  # Will be used to pass the image size around
        self.ploty: Optional[np.ndarray] = None   # Will hold a list of evaluation points for the lane polynomial


class Lane:
    """Class for finding one lane."""

    def __init__(self, config: LaneFindingConfig):
        self.cfg = config
        self.is_stable: bool = False       # True if last frame resulted in a lane detection
        self.coefficients: Optional[np.ndarray] = None   # Coefficients of the lane polynomial fit (2nd order)
        self.number_of_unstable_frames: int = config.max_unstable_frames  # How many frames since last stable frame
        self.x_pixels_sliding_window = []
        self.y_pixels_sliding_window = []

    @property
    def is_valid_coefficients(self):
        return self.coefficients is not None and len(self.coefficients) == 3

    def poly_eval(self, y: Union[int, np.ndarray],
                  coeffs: Optional[np.ndarray] = None) -> Optional[Union[np.int32, np.ndarray]]:
        """Evaluates the polynomial on the given points or list of points.
        If no coefficients are supplied, use the objects coefficients

        Returns a point, a list of points, or None if the there is no polynomial"""
        if coeffs is not None:
            coefficients = coeffs
        elif self.is_valid_coefficients:
            coefficients = self.coefficients
        else:
            coefficients = None

        if coefficients is not None:
            x = coefficients[0] * y ** 2 + coefficients[1] * y + coefficients[2]
            return np.int32(np.round(x))
        else:
            return None

    def curvature(self, y_px: int) -> Optional[float]:
        """If there is a polynomial fit to the lane, return the curvature (1/radius), otherwise return none.
        The input is in pixels but the output is in 1/m."""
        if self.is_valid_coefficients:
            a, b, _ = self.coefficients
            a = (self.cfg.mx / self.cfg.my ** 2) * a
            b = (self.cfg.mx / self.cfg.my) * b
            y = self.cfg.my * y_px
            return (2 * np.abs(a)) / ((1 + (2 * a * y + b) ** 2) ** (3 / 2))
        else:
            return None

    def find_points_sliding_window(self, img: np.ndarray, initial_point: Optional[int]) -> Tuple[List[int], List[int]]:
        """Uses a sliding window to select the group of points that are part of the lane.
        Returns a list of x coordinates and y coordinates."""
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0] // self.cfg.n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        # If there is a detection in memory, use it as the starting point, otherwise, use the initial_point
        if initial_point is None:
            if self.is_valid_coefficients:
                biggest_y = img.shape[0] - 1
                x_current = self.poly_eval(biggest_y)
            else:
                x_current = np.int(np.round(img.shape[1] / 2))
        else:
            x_current = initial_point

        # Create empty lists to receive lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(self.cfg.n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_x_low = x_current - self.cfg.margin
            win_x_high = x_current + self.cfg.margin

            good_inds = np.where((nonzerox >= win_x_low) &
                                 (nonzerox < win_x_high) &
                                 (nonzeroy >= win_y_low) &
                                 (nonzeroy < win_y_high))[0]

            # Append these indices to the lists
            lane_inds.extend(good_inds)

            if len(good_inds) > self.cfg.minpixels:
                x_current = np.int(np.mean(nonzerox[good_inds]))
        # Extract left and right line pixel positions
        x_pixels = nonzerox[lane_inds]
        y_pixels = nonzeroy[lane_inds]
        self.x_pixels_sliding_window = x_pixels
        self.y_pixels_sliding_window = y_pixels

        return x_pixels, y_pixels

    def fit_lane(self, img: np.ndarray, initial_point: int = None) -> None:
        """Uses the sliding window method to find a 2nd deg polynomial for the lane"""
        x_pixels, y_pixels = self.find_points_sliding_window(img, initial_point)
        unstable = True
        if len(x_pixels) > 10:  # Minimum number of pixels to have a non-trivial fit
            candidate = np.polyfit(y_pixels, x_pixels, 2)
            eval_start = np.int(np.round(2 * len(self.cfg.ploty) / 3))  # Check bottom third of image
            xs = self.poly_eval(self.cfg.ploty[eval_start:], candidate)
            unstable = (np.any(xs <= 0) or
                        np.any(xs >= self.cfg.image_size[1]))  # Unstable if polynomial outside of image

        if unstable:
            self.number_of_unstable_frames += 1
            if self.number_of_unstable_frames >= self.cfg.max_unstable_frames:
                self.coefficients = None
                self.is_stable = False
        else:
            self.coefficients = candidate
            self.is_stable = True
            self.number_of_unstable_frames = 0


class LaneFinding:
    """Responsible for finding both lanes in an image and holding the state between frames while lane finding in a
    movie.

    It assumes that there are only two lane marks, one on each side."""
    def __init__(self, config: LaneFindingConfig, use_prev_state: bool = True):
        self.cfg = config
        self.l_lane = Lane(self.cfg)
        self.r_lane = Lane(self.cfg)
        self.hist: Optional[np.ndarray] = None
        self.lane_idxs: Optional[Sequence[int]] = None
        self.dist_center_m: float = 0.0
        self.curvature: float = 0
        self.lane_center_px: int = 0
        self.use_prev_state = use_prev_state    # If true, will try to use previous detection to bootstrap current

    @property
    def lanes(self) -> Tuple[Lane, Lane]:
        return self.l_lane, self.r_lane

    @property
    def canonical_curvature_radius(self):
        """Returns the radius of curvature (instead of curvature)"""
        if abs(self.curvature) < 1e-3:  # Basically a straight line
            return float('inf')
        else:
            return 1 / self.curvature

    @property
    def all_lanes_stable(self):
        return self.l_lane.is_stable and self.r_lane.is_stable

    def find_from_hist(self, img: np.ndarray) -> Tuple[Sequence[int], np.ndarray]:
        """Uses the histogram technique to find the center of the left and right lanes.

        `lane_width`: the expected lane width, in pixels, on the bottom of the image. It is used to smooth the
        histogram.

        Return: sequence with the x position of the two highest hist values, list with the histogram
        """
        img_size = img.shape[:2]
        bottom_half = img[(img_size[0] // 2):, :, ]  # The part nearest to the car should be straighter
        histogram = np.sum(bottom_half, axis=0)
        smoothing_kernel = np.ones((self.cfg.hist_smoothing,)) / self.cfg.hist_smoothing
        smoothed_hist = np.convolve(histogram, smoothing_kernel, mode='same')

        # Now find the two highest local maxima that are separated by min_lane_separation_px
        local_max: List[Tuple[int, int]] = []  # (value, index)

        # Step 1: find local maxima
        for i, (p, c, n) in enumerate(zip(np.append(smoothed_hist[0:1], smoothed_hist),
                                          smoothed_hist,
                                          np.append(smoothed_hist, smoothed_hist[-1:])[1:])):
            if p <= c > n:
                local_max.append((c, i))

        # Step 2: For each local maximum, check if it is the absolute maximum within self.cfg.min_lane_separation_px
        max_in_lane_width: List[Tuple[int, int]] = []  # (value, index)
        i = 0
        while i < len(local_max):
            point = local_max[i]
            if i == (len(local_max) - 1):
                max_in_lane_width.append(point)
                break
            larger_found = False
            for next_point in local_max[i + 1:]:
                i += 1  # Every point we check in this loop we'll not have to check again on the outer loop
                if next_point[1] > (point[1] + self.cfg.min_lane_separation_px):
                    break   # Reached the farthest point to search
                if next_point[0] >= point[0]:
                    larger_found = True
                    break
            if not larger_found:
                max_in_lane_width.append(point)

        # Step 3: Get the two largest points on the hist
        if len(max_in_lane_width) >= 2:
            two_max = sorted(max_in_lane_width)[-2:]
            max_indexes = (two_max[0][1], two_max[1][1])
            max_indexes = sorted(max_indexes)   # Sort so left is always before right
            return max_indexes, smoothed_hist
        else:
            raise Exception('Did not find two maximums in histogram: %d.' % len(local_max))

    def find_lanes(self, img: np.ndarray) -> np.ndarray:
        """Finds the left and right lane in the image"""
        # Update the cached values
        if self.cfg.image_size is None:
            self.cfg.image_size = img.shape
        if self.cfg.ploty is None:
            self.cfg.ploty = np.linspace(0, img.shape[0] - 1, img.shape[0], dtype=np.int)

        ploty = self.cfg.ploty
        for lane in self.lanes:
            if lane.is_stable:
                lane.fit_lane(img)

        # Check that the lanes are all stable and that they are parallel enough
        is_detection_good = False
        if self.l_lane.is_stable and self.r_lane.is_stable:
            pos_top_px = []
            pos_bottom_px = []
            for lane in self.lanes:
                pos_px = lane.poly_eval(0)
                pos_top_px.append(pos_px)

                pos_px = lane.poly_eval(img.shape[0])
                pos_bottom_px.append(pos_px)
            dist_top_px = pos_top_px[0] - pos_top_px[1]
            dist_bottom_px = pos_bottom_px[0] - pos_bottom_px[1]
            is_detection_good = abs(dist_top_px - dist_bottom_px) < (self.cfg.max_parallel_diff / self.cfg.mx)

        if not is_detection_good or not self.use_prev_state:
            lane_idxs, hist = self.find_from_hist(img)
            self.lane_idxs = lane_idxs
            self.hist = hist
            lane_width_px = abs(lane_idxs[0] - lane_idxs[1])
            self.cfg.mx = self.cfg.lane_width / lane_width_px

            for lane, idx in zip(self.lanes, lane_idxs):
                lane.fit_lane(img, idx)

        # Generate x and y values for plotting
        lanes_fit_x = []
        for lane in self.lanes:
            fitx = lane.poly_eval(ploty)
            if fitx is None:
                # Avoids an error if either `left` or`right_fit` are still none or incorrect
                print('The function failed to fit a line!')
                fitx = 1 * ploty ** 2 + 1 * ploty
            lanes_fit_x.append(np.round(fitx))

        # Get the canonical curvature from the highest support lanes
        if self.hist[self.lane_idxs[0]] > self.hist[self.lane_idxs[1]]:
            lane_for_curvature = self.lanes[0]
            fit_x_for_curvature = lanes_fit_x[0]
        else:
            lane_for_curvature = self.lanes[1]
            fit_x_for_curvature = lanes_fit_x[1]
        # Find the curvature on the farthest point that is still on the image
        valid_xs_for_curvature = (fit_x_for_curvature >= 0) & (fit_x_for_curvature < img.shape[1])
        ploty_for_curvature = ploty[valid_xs_for_curvature]
        self.curvature = lane_for_curvature.curvature(ploty_for_curvature[0])

        # Use the polynomials to find the lane center
        if self.all_lanes_stable:
            self.lane_center_px = int(np.round((self.lanes[0].poly_eval(ploty[-1]) + self.lanes[1].poly_eval(ploty[-1])) /
                                               2.0))
        self.dist_center_m = (self.lane_center_px - (img.shape[1] / 2)) * self.cfg.mx

        # Colors in the left and right lane regions and the area between them
        out_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        out_img[:, :, 0] = 255 * img
        # Plot the points used to find each lane
        for i, lane in enumerate(self.lanes):
            color = [196, 109, 255] if i == 0 else [99, 255, 234]
            out_img[lane.y_pixels_sliding_window, lane.x_pixels_sliding_window, ...] = color

        # Color the lane area
        valid_xs_for_lane = ((lanes_fit_x[0] >= 0) &
                             (lanes_fit_x[0] <= img.shape[1]) &
                             (lanes_fit_x[1] >= 0) &
                             (lanes_fit_x[1] <= img.shape[1])
                             )
        # out_img[ploty, lanes_fit_x[0]] = [0, 0, 255]
        # out_img[ploty, lanes_fit_x[1]] = [0, 0, 255]
        points_to_plot = zip(ploty[valid_xs_for_lane],
                             lanes_fit_x[0][valid_xs_for_lane],
                             lanes_fit_x[1][valid_xs_for_lane])
        for y, x_start, x_end in points_to_plot:
            out_img[y, x_start:x_end, 1] = 200
        return out_img

    def get_find_lanes_as_processor(self,):
        """Returns a ``SimpleImageProcessor`` that executes the method ``find_lanes``"""
        return WrappedImageProcessor(self.find_lanes, 'Find lanes')


class OverlayHist(MergingImageProcessor):
    """Extends the original image with a drawing of the histogram used to find lane markings"""
    def __init__(self, lane_finder: LaneFinding,
                 source_name: Optional[str] = None,
                 height: int = 150,
                 fg: Iterable[int] = (199, 219, 249),
                 bg: Iterable[int] = (0, 0, 0),
                 lane_center_color: Iterable[int] = (255, 0, 255)):
        MergingImageProcessor.__init__(self, 'Overlay Histogram')
        self.lane_finder = lane_finder
        self.source_name = source_name
        self.height = height
        self.fg = fg
        self.bg = bg
        self.lane_center_color = lane_center_color

    def process(self, img: np.ndarray) -> np.ndarray:
        """If a single input is used, assume it is the same image used to extract the """
        return self.merge(img, img)

    def merge(self, img: np.ndarray, lane_colors: np.ndarray) -> np.ndarray:
        lane_idxs, hist = self.lane_finder.find_from_hist(lane_colors)
        lane_center_px = self.lane_finder.lane_center_px

        img_size = img.shape
        result_size = list(img_size)
        result_size[0] += self.height
        img_with_hist = np.zeros(result_size)
        img_with_hist[:, :, ...] = self.bg
        img_with_hist[:img_size[0], :, ...] = img

        # Prepare the histogram
        scaled_hist = np.uint8(np.round(hist / np.max(hist) * self.height))
        for i in range(self.height):
            row = img_size[0] + i
            img_with_hist[row, scaled_hist >= i, ...] = self.fg
            img_with_hist[row, lane_center_px, ...] = self.lane_center_color
        for i in lane_idxs:
            img_with_hist[-self.height:, i, ...] = (255, 0, 0)
            img_with_hist[-self.height:, i - 1, ...] = (255, 0, 0)
            img_with_hist[-self.height:, i + 1, ...] = (255, 0, 0)
        return np.uint8(img_with_hist)


class WriteInfo(SimpleImageProcessor):
    """Write measured information on the image"""
    def __init__(self, lane_finder: LaneFinding):
        SimpleImageProcessor.__init__(self, 'Write Info')
        self.lane_finder = lane_finder

    def process(self, img: np.ndarray) -> np.ndarray:
        img_out = img.copy()
        rad_curve_string = 'Radius of curvature: %.0f m' % self.lane_finder.canonical_curvature_radius
        dist_from_center_side = 'left' if self.lane_finder.dist_center_m <= 0 else 'right'
        lane_pos_string = 'Vehicle if %0.2f m %s of center' % (abs(self.lane_finder.dist_center_m),
                                                               dist_from_center_side)
        cv2.putText(img_out,
                    rad_curve_string,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.putText(img_out,
                    lane_pos_string,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        return img_out


def add_overlay(processor: Optional[ImageProcessor]):
    """Add `processor` to a pipeline with an overlay"""
    if processor is None:
        return processor

    cal = CameraCalibration()
    cal.load()
    undistorter = cal.get_undistort_as_processor()
    undistorter.checkpoint_name = 'start'

    pipeline = undistorter.add(processor)
    pipeline.add(To3Channel())
    pipeline.add(CombineImages(source_name='start', β=0.5))
    pipeline.name = processor.name
    return pipeline


def test_in_all_images_with_overlay(processors: np.ndarray, folder='test_images', output_prefix: str = ''):
    processors = np.vectorize(add_overlay)(processors)
    test_in_all_images(processors, folder, save_images=True, output_prefix=output_prefix)


def test_in_all_images(processors: np.ndarray, folder='test_images', cmap=None,
                       save_images: bool = True,
                       output_prefix: str = '',
                       out_folder: str = 'output_images',
                       ):
    nrows, ncols = processors.shape
    for fname in glob.glob(folder + os.sep + '*.jpg'):
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18))
        fig.suptitle(fname)
        fig.tight_layout()
        img = mpimg.imread(fname)
        if nrows == 1 and ncols == 1:
            loop_vars = [(processors.flatten()[0], axes)]
        else:
            loop_vars = zip(processors.flatten(), axes.flatten())
        for processor, axis in loop_vars:
            if processor is None:
                continue
            axis.imshow(processor.process(img), cmap=cmap)
            axis.set_title(processor.name, fontsize=12)

        plt.subplots_adjust(left=0., right=1., top=0.9, bottom=0.)
        if save_images:
            _, out_file_name = os.path.split(fname)
            out_file_name = output_prefix + out_file_name
            out_full_name = os.path.join(out_folder, out_file_name)
            plt.savefig(out_full_name, bbox_inches='tight')
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
    plt.show()


def test_transforms():
    s_t = SChannelThreshold()
    r_t = RChannelThreshold()
    # h_t = HChannelThreshold()
    c_t = SRThreshold(s_t, r_t)
    f_dir = DirectionalThreshold(kernel=15, thresh=(radians(30), radians(60.0)))
    f1x = SobelThreshold(orient='x', kernel=7, thresh=(15, 70))
    f1y = SobelThreshold(orient='y', kernel=7, thresh=(15, 70))
    f_mag = MagnitudeThreshold(kernel=9, thresh=(15, 70))
    f4 = CombinedAnalysis(proc_color=c_t, proc_x=f1x, proc_y=f1y, proc_dir=f_dir, proc_mag=f_mag)
    # unitary = SimpleImageProcessor()
    test_in_all_images_with_overlay(np.array([
        # [s_t, r_t],
        [c_t, f_dir],
        [f1x, f1y],
        [f_mag, f4],
    ]), output_prefix='test_transforms_')


def test_perspective_transform():
    cal = CameraCalibration()
    cal.load()
    undistorter = cal.get_undistort_as_processor()

    pt = PerspectiveTransform()
    transform = pt.get_transform_as_processor()

    pl_no_transform = undistorter.add(DrawPoly(pt.src))
    pl_no_transform.name = 'Reference for perspective transform'
    pl_transform = undistorter.add(DrawPoly(pt.src)).add(transform).add(DrawPoly(pt.dst, color=(122, 28, 255)))
    pl_transform.name = 'Transformed image'

    funcs = np.array([[pl_no_transform], [pl_transform], ])
    test_in_all_images(funcs,
                       output_prefix='test_perspective_')


def test_pipeline(do_overlay: bool = True):
    pipeline = create_full_pipeline(debug=True, single_images=True, do_overlay=do_overlay)
    funcs = np.array([[pipeline, ], ])
    output_prefix = 'test_full_pipeline_' if do_overlay else 'test_pipeline_raw_'
    test_in_all_images(funcs, output_prefix=output_prefix)


def create_full_pipeline(debug: bool = True, single_images: bool = False, do_overlay: bool = True) -> ImageProcessor:
    cal = CameraCalibration()
    cal.load()

    lane_finder = LaneFinding(LaneFindingConfig())
    lane_finder.use_prev_state = not single_images   # Previous state makes no sense for single images
    pt = PerspectiveTransform()

    # Assemble the pipeline
    undistorter = cal.get_undistort_as_processor()
    undistorter.checkpoint_name = 'calibrated'  # We'll use it later to recombine the images
    pipeline = undistorter.add(pt.get_transform_as_processor())
    image_threshold = CombinedAnalysis()
    image_threshold.checkpoint_name = 'lane_detection'
    pipeline.add(image_threshold)
    pipeline.add(lane_finder.get_find_lanes_as_processor())
    if do_overlay:
        pipeline.add(pt.get_restore_as_processor())
        pipeline.add(CombineImages(source_name='calibrated'))
    pipeline.add(WriteInfo(lane_finder))
    if debug:
        pipeline.add(OverlayHist(lane_finder, source_name='lane_detection'))

    return pipeline


def run_full_pipeline(video_file='project_video.mp4', out_folder='./output_videos'):
    video = VideoFileClip(video_file)   # .subclip(20, 25)
    pl = create_full_pipeline(debug=False, single_images=False)
    processed_video = video.fl_image(pl.process)
    processed_video.write_videofile(out_folder + os.sep + video_file, audio=False)


def print_help():
    """Print the command line help"""
    print("""
Usage: proj.py [--test [cal | transf | perspective | pipeline]] [cal | FILE]
Process a movie and generate another movie with the lane marking detection info.

Usage for movies (results always saved on folder ./output_movies):
    proj.py             # Process the file project_video.mp4
    proj.py <file.mp4>  # Process the file
    
Usage for saving calibration information:
    proj.py cal  # Generates calibration information and save to calibration.pk

Usage for testing (images saved to ./output_images):
    proj.py --test cal                    # Test the calibration step
    proj.py --test transf                 # Test the lane pixel detection step
    proj.py --test perspective            # Test the perspective transformation step
    proj.py --test pipeline [no_overlay]  # Test the full pipeline on static images.
""")


if __name__ == '__main__':
    import sys

    # FIXME: Use argparse to implement it
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        if not len(sys.argv) > 2:
            print('--test what?')
            print_help()
            exit(1)
        to_test = sys.argv[2]
        if to_test == 'cal':
            test_calibration()
        elif to_test == 'transf':
            test_transforms()
        elif to_test == 'perspective':
            test_perspective_transform()
        elif to_test == 'pipeline':
            do_overlay = (len(sys.argv) == 3) or sys.argv[3] != 'no_overlay'
            test_pipeline(do_overlay=do_overlay)
        else:
            print('Option not understood: %s' % (' '.join(sys.argv[1:])))
            print_help()
            exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == 'cal':
        run_calibration_and_save()
    else:
        video_filename = sys.argv[1] if len(sys.argv) > 1 else 'project_video.mp4'
        print('Running full pipeline in file %s' % video_filename)
        run_full_pipeline(video_file=video_filename)
