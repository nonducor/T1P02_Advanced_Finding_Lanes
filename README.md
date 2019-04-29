## Writeup Advanced Lane Finding

_Note_: This code requires Python 3.6.

[//]: # (Image References)

[cal_ex1]: ./output_images/calibration3.jpg "Checker pattern detection"
[cal_ex2]: ./output_images/compare_calibration.png "Undistorted"
[transf1]: ./output_images/test_transforms_test2.jpg "Transformation test"
[perspective1]: ./output_images/test_perspective_straight_lines2.jpg "Perspective transform"
[pipe1]: ./output_images/test_full_pipeline_straight_lines2.jpg "Full pipeline straigh lane"
[pipe2]: ./output_images/test_full_pipeline_test2.jpg "Full pipeline curve"
[pipe_raw1]: ./output_images/test_pipeline_raw_straight_lines2.jpg "Full pipeline straight lane raw"
[pipe_raw2]: ./output_images/test_pipeline_raw_test2.jpg "Full pipeline curve  raw"
[video1]: ./output_videos/project_video.mp4 "Video"


### Code overview

The code is divided in 4 files:
    proj.py: main file
    calibration.py: contains a class to calculate and record the calibration information
    image_analysis.py: contains classes and experimentation related to the image pre-processing for lane detection
    util.py: contains a set of classes related to a processing pipeline description

#### `util.py`    
This file contains a basic infrastructure for defining a processing pipeline, able to do some simple branching. The base class is `ImageProcessor`, with 3 main subclasses:

1. `SimpleImageProcessor`: wraps a simple image processing operation. The processing operation is done by the method `process`, that receives an image as input and outputs another image. It also contains the convenience method `add`, the returns a processing pipeline with this image processor as the first element and the added image processor as the second one.
1. `CompositeImageProcessor`: allows simple processing pipelines to be defined. A basic processing pipeline is just a chaining of image transformations, with the option of very simple branching (that was all that was needed for this project). Branching is done by setting some image processors as _checkpointing_, meaning that their output will be remembered (attached to a name) for later usage.
1. `MergingImageProcessor`: allows definition of processors that consume two images and output a single image. The first image comes from whatever processor is connected before them on the chain, while the second image is a "checkpointed" image that has to be named during the pipeline configuration.

Besides these 4 classes, some useful processors are defined:
1. `WrappedImageProcessor`: allows wrapping a simple function and transforming it into an image processor. Useful to wrap calibration and perspective transformation/restoration methods from other classes.
1. `CombineImages`: a `MergingImageProcessor` that overlays two images.
1. `To3Channels`: a useful function to "lift" a single channel image to a 3 channel image (optionally rescaling its color from 0 to 1 to another range). It does not change images that already have 3 channels, so it can be added to pipelines during experimentation to increase flexibility.

#### `image_analysis.py`
This file contains several sub-classes of `SimpleImageProcessor`s with image transformations useful for finding lanes. Not all transformations that are defined in this file were used on this project (they were used during experimentation)

#### `calibration.py`
Contains a class to calculate camera calibration, save and load the coefficients and apply the undistortion to any image (including a `SimpleImageProcessor` producer).

#### `perspective.py`
Contains a class called `PerspectiveTransform` that wraps around the cv2 methods to calculate perspective transformation matrices and apply perspective transformations to ensure that the transformation and "restoration" of the perspective use the same parameters.

#### `proj.py`
Main file for this project. Contains the lane finding algorithm, perspective transformation, test pipelines and command line argument parsing.

----

### Camera Calibration
The code for calibration is on file `calibration.py`. On the method `calibrate` of the class `CameraCalibration`, a list of file names can be passed. Each file is expected to contain a checkerboard pattern of `nx` x `ny` corners (in this case, 9x6), acquired with the same camera. The object points are assumed to be in a flat plane at z=0, with the (x, y) coordinates coincident with the corner coordinates (e.g. (1,1), (1,2) etc).

When the code is run, `objpoints` is appended with a copy of these ideal object points while `imgpoints` is appended with the chessboard corners found on the image (only when the corners are successfully found). The figure below shows an example of corners being detected on an image. 

![Corner detection][cal_ex1]

Finally, the `cv2.calibrateCamera()` function is used to calculate the distortion parameters. These parameters can be saved to a file (through the method `save`) and loaded in the future (through method `load`). The method `undistort` just wraps the function `cv2.undistort`, providing the calibration parameters found through `calibrate` or loaded from a file. An undistorter image processor can be obtained from the method `get_undistort_as_processor`, for use within a processing pipeline.

The figure below shows a comparison of the distorted (left) and undistorted (right) version of an image:

![Image undistortion][cal_ex2]

The calibration function can be accessed by calling the script as:

    python3 proj.py cal
to save the calibration data to the file `calibration.pk` or

    python3 proj.py --test cal
if it is desired to show the intermediate steps on the screen and save the images used on this report.

----

### Perspective transformation

The class `PerspectiveTransform` in file `perspective.py` wraps the perspective transformation operations. To find the `src` points, trial and error was used. The goal was to map a stripe that is slightly larger than a straight lane to a bird's eye view of the lane. The mapping was found out by using a calibrated image and then testing on all images in `test_images` folder.

The class constructor calculates the transformation matrices and the class offers convenience methods to transform and restore an image (restoration is useful for projecting back the results over the original image). It also can provide both methods wrapped in `ImageProcessor`s for usage in a pipeline.

The parameters used on the perspective transformation were hand-picked, by analysing the straight line pictures, and are the following:

| Source (x, y) | Destination (x, y) | 
|:-------------:|:------------------:| 
| 224, 719      | 300, 720           | 
| 619, 440      | 300, 0             |
| 682, 440      | 980, 0             |
| 1100, 719     | 980, 720           |

To test the results, the testing function can be called with:

    python3 proj.py --test perspective
    
The figure below shows the result for the one of the figures. The blue parallelogram on the source figure shows the `src` points. On the transformed image, the purple rectangle (drawn using the `dst` points) can be seen superimposed to the blue rectangle, showing that the transformation was successful. Also, it can be seen that both lane markings are approximately parallel to the edges of the purple rectangle, showing that no unwanted distortions were introduced by the transformation.

![Perspective transform][perspective1]


### Lane pixel identification
----

The identification of the most adequate image transformation for lane detection was done using a lot of trial and error. To facilitate this trial and error, a small test infrastructure was developed.

First, on the file `image_analysis.py` several processing elements were implemented, each with either a simple transformation or a simple combination of transformations. Then, several tries are assembled in a table and plotted for comparison, using the test images. This test can be accessed by:
    
    python3 proj.py --test transf
    
One example of the pipeline result can be seen in the figure below.

![Transformed images][transf1]

The transformations are overlaid in red over the original image to aid in visually checking if the detected edges are the correct ones. The testing pipeline can be found in function `test_transform` on file `proj.py`. The images are first undistorted (the undistortion step of the pipeline is added on the function `test_in_all_images`). This function receives an array of transformations and applies these transformation for all test images, showing the result for each image.

The chosen transformation is a combination of color-base and gradient based transformations. The processor implementing this transformation is on file `image_analysis.py`, in class `CombinedAnalysis`. This class implements a combined binary detection, using other detectors, according to the following formula:

    Part_of_lane_marking = (<S Threshold> and <R Threshold>) or
                            ((<gradx> and <grady>) or (<mag_binary> and <dir_binary>))

The parameters were all found by trial and error. The transformations are:

1. Color thresholding: two color thresholding are used, one on the "color" S on the HLS space and the other on the color red on the RGB space. The class implementing the combined S & R detection is `SRThreshold`. It combines the results of two simple thresholdings:
    1. On the HLS space, all pixels with saturation values above 90 are selected. This usually does a good job in good lighting conditions, but tends to select a lot of pixels when there are shadows involved. This is implemented by class `SChannelThreshold`.
    1. To help alleviate the problem, this detection is "anded" with a red threshold. All pixels with a valor of red above 180 are selected. This helps, for example, remove saturated colors that are not white. Implemented in `RChannelThreshold`.
    
1. The <grad{x,y}> Sobel processors apply a threshold over the gradient on the x and y directions. If the gradient is withing a given threshold for both cases, then it is considered accepted. For this combination, a smoothing kernel of 7 pixels is used and a threshold of (15, 70) is used. The limits on the threshold are inclusive. In class `SobelThreshold`.

1. The magnitude processor calculates the absolute magnetude of the gradient (using Sobel derivatives on the x and y directions) and selected the pixels within a given threshold. The threshold is scaled in the sense that the highest gradient in the image is used to set the value of 255. The threshold used on this application was (15, 70), with a smoothing kernel of 9 pixels. In class `MagnitudeThreshold`.

1. The directional processor uses the Sobel derivatives to calculate the absolute direction of the derivativa and selects those points that are within a given angular range. The intuition is that lanes will mostly be on an almost "vertical" angle on the image, so looking for high derivatives perpendicular to the expected direction of the lanes should be a good detector os lane marking frontiers. The main issue with this detector is that it is quite noisy (as can be seen on the figure above). Thus, it is combined with the threshold processor to detect only the best possibilities. In class `DirectionalThreshold`.



----
### Pipeline
#### Basic Pipeline
The pipeline is composed of the following steps, in order (it is created by the function `create_full_pipeline` in `proj.py`):
1. Apply image undistortion (from camera calibration) and checkpoint the result as `calibrated` to be combined in the future with the image processing steps.
1. Do a perspective transformation to get a bird eye view from the relevant parts of the image.
1. Apply the relevant image transformation to get the basis for lane detection. Checkpoint it on the pipeline as `lane_detection`.
1. Run the main lane finding algorithm on the resulting image (from a `LaneFinding` instance)
1. Restore the perspective of the image to allow overlaying it to the original image
1. Combine the processed image to the original `calibrated` image
1. Write the calculated radius of curvature and offset from lane center to the image
1. If debugging information is required, plot the histogram and calculated lane center on the bottom of the image. This is not the same information as used by the `lane_detection` instance, but it is useful for understanding how stable the lane detection step is (if one of the sides of the histogram is too small or too wide, it may indicate a challenging frame).


Since the full pipeline is represented as an image processor, it can be applied to the test images or to a movie easily. When applied to individual images, it is importante to turn off the "statefulness" of the lane detection, so it doesn't use the previous detections to bootstrap the current one. Also, when images are used the reverse perspective transform is not applied and the image is not overlaid back to the camera image. This allows an easier debugging of issues.


### Lane finding algorithm

The lane finding alorithm is implemented by 3 classes, all on file `proj.py`:
* `LaneFindingConfig`: contains useful parameters for the algorithms. Is a data holder class and it provides no logic. Of notice, the geometric parameters of the lane are: 3.7 m of width (from US standard lanes) and 40.0 m to the "horizon" (the end of the perspective transform window). This was measured considering that, according to the US standard, the distance between the beginning of each segment in the broken lane markings is 12 m and slight less then 4 markings can be seen (the perspective transformed images are especially good for measuring it).

* `Lane`: this class holds data and functions useful for, given a starting point for a lane marking, to find it on an image and generate a 2nd order approximation for it. It also implements some crude estimation of the stability of the detection, to be used to select between simpler or more complex algorithms for lane detection.

* `LaneFinding`: is the main entry point for the lane detection algorithm.

The main entry point for the lane finding pipeline is the method `find_lanes` on the `LaneFinding` class. This method receives an image (expected to be binary, result of the processing by the image processor explained before) and tries to find the two lane markings using the techniques shown in class. The basic flow of this method is:
1. If the detections are stable (basically, if the previous detections were successful and the detected lines are parallel up to 1 m), delegate the lane marking detection to a class specialized for single lane marking detection. This class will uses the sliding window technique shown in class to find the pixels that are part of the lane and fit a second order polynomial to it. The only change from what was shown in class is a slightly vectorization of the code that improved performance.
1. If the detection was not stable (either because the single lane detectors indicated it or because the detection resulted in non-parallel lanes), use the histogram technique to try to find out the lane axis and use again the techniques shown in class to fit the polynomial
1. With the polynomial coefficients, generate a set of x points for each lane marking.
1. Calculate the curvature on the farthest point possible from ego (curvature calculation is done by method `Lane.curvature`). This point is selected to be the most distant point from ego that is still on the perspective corrected image (see [this image](#Application-to-single-images) for an example). The lane marking used to calculate the curvature is the one that has the "biggest support", in the sense that it had the highest peak on the histogram. The intuition behind this decision is that this usually is related to the continuous lane, when one lane marking is broken and the other continuous. Usually this has the most precise polynomial fit. The drawback is that, when an image feature such as shadows creates a lot of noise, this side may appear as having a higher support. Note that curvature (1/radius) is used instead of the radius of curvature. This definition is more convenient because straight lines have 0 curvature but infinite radius of curvature.
1. Also using the polynomials, the position in relation to the lane center is found, by using the evaluation on the point nearest to the car.
1. Finally, the image is changed to add information about the detection. See the [section on single lanes below](#Application-to-single-images) for the meaning of each color.

The histogram technique is similar to the one used in class. It is implemented by method `LaneFinding.find_from_hist`. The differences are:
1. The raw histogram is smoothed by a moving average function of 30 pixels. This  reduces the number of local maximums.
1. The two selected peaks have to be at a minimum distance from each other (this avoids finding two peaks that are very close to each other). The minimum distance is set to 400 pixels.

All parameters for the histogram detection are configurable from `LaneFindingConfig`.



#### Application to single images

By using the command below, the full pipeline can be tested on the static images:

    python3 proj.py --test pipeline

If the `no_overlay` option is used, the information is not overlaid back on the original image. This is helpful in debugging.

You can see an example of the pipeline being applied to a single image below.

![Pipeline single image straight lane][pipe1]

The histogram is displayed on the bottom part of the image, for debugging purposes. The histogram shows the two peaks that were identified with a lane (in red) and the identified center of the lane (in purple).

On the image, the color convention is:
* Red: pixels selected by the image processing pipeline
* Light purple: pixels selected as part of the left lane marking by the lane finding algorithm 
* Light blue: pixels selected as part of the right lane marking
* Green: area of the lane, drawn by tracing a line from the polynomial interpolated on the left and on the right side.

An example of a curving lane can be seen in:

![Pipeline single image curving lane][pipe2]

Analyzing the images without projecting back to the scene is quite helpful also. The image below is the same case as the curving strip above. It is easier to see which pixels in each lane marking are being used for interpolation:

![Pipeline single image curving lane][pipe_raw2]


#### Application to a movie

As explained above, an `ImageProcessor` has a single entry point, the method `process`. To process the images on a movie, the bound method is passed as the argument of `VideoFileClip.fl_image` for the video clip loaded from the image.

![Video of the lane detection][video1]

On this movie, the detection is mostly stable, with some issues with the shadows.

---

### Discussion

The approach I implemented will likely work on dry roads, on sunny days and with well-painted lane markings. Nevertheless, the method is very brittle for changes in lighting and lane quality. The method proved very unreliable on the challenge videos.

Interesting though, I believe the method will be somewhat resistant to cars crossing in front of ego, as they did not generate strong detections (on the video, you can see some red spikes around the black car, which are all the noise that the lane finding algorithm would have to use.

Also, I did not implement any kind of dynamic assumption on the algorithm, so fast changes in lane position and curvature are "accepted" by the method. Some good improvements and good noise rejection could come from this method. The drawback is that it fails in situations were the lanes do change faster than a simple car dynamical model would expect, such as in crossings and access ramps.

