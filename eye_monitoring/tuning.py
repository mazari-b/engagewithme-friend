from __future__ import division
import cv2
from .core_eye import core_eye


class Tuning(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.unbinarised_frames = 20
        self.left_thr = []
        self.right_thr = []

    def tuning_finished(self):
        """Boolean to indicate if tuning has finished"""
        return len(self.left_thr) >= self.unbinarised_frames and len(self.right_thr) >= self.unbinarised_frames

    def eye_threshold(self, RorL):
        """eye_threshold: threshold for specified eye"""
        if RorL == 0:
            return int(sum(self.left_thr) / len(self.left_thr))
        elif RorL == 1:
            return int(sum(self.right_thr) / len(self.right_thr))

    @staticmethod
    def size_of_i(eye_image):
        """Returns size of iris taking up the eye"""
        eye_image = eye_image[5:-5, 5:-5]
        h, w = eye_image.shape[:2]
        unbinarized_px = h * w
        unbinarized_blks = unbinarized_px - cv2.countNonZero(eye_image)
        return unbinarized_blks / unbinarized_px

    @staticmethod
    def optimal_point(image_eye):
        """Calculates the optimal threshold to binarize the
        frame for the given eye."""
        standard_size = 0.48
        checked = {}

        for potential_value in range(5, 100, 5):
            frame_is = core_eye.preprocess_image(image_eye, potential_value)
            checked[potential_value] = Tuning.size_of_i(frame_is)

        optimal_value, iris_size = min(checked.items(), key=(lambda p: abs(p[1] - standard_size)))
        return optimal_value

    def improve(self, frame_of_eye, LorR):
        """Improvement of calibration using frame of eye """
        binarisation_point = self.optimal_point(frame_of_eye)

        if LorR == 0:
            self.left_thr.append(binarisation_point)
        elif LorR == 1:
            self.right_thr.append(binarisation_point)
