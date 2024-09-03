from __future__ import division
import cv2
from .core_eye import core_eye

# Finds the best limit for tuning the algorithm
class Tuning(object):
    def __init__(self):
        self.l_limit = []
        self.r_limit = []
        self.unbinarised = 20
        
    # Mathematical calculation of an eye's threshold
    def average_th(self, RorL):
        if RorL not in (0, 1):
            raise ValueError("RorL must be 0 (left eye) or 1 (right eye).")
        limit_list = self.l_limit if RorL == 0 else self.r_limit
        # Check to prevent division by zero if the list is empty
        if len(limit_list) == 0:
            raise ValueError("Limit list is empty, cannot calculate average.")
        # Calculate and return the average as an integer
        return int(sum(limit_list) / len(limit_list))

    # Boolean return: indication if tuning has finished or not
    def tuning_finished(self):
        # Check if elements in l_limit and r_limit reached the necessary threshold
        left_eye_tuned = len(self.l_limit) >= self.unbinarised
        right_eye_tuned = len(self.r_limit) >= self.unbinarised
        # If both have, True will be returned
        return left_eye_tuned and right_eye_tuned

    # Calculation of best limit to binarise the eye frame
    @staticmethod
    def optimal_point(image_eye):
        # Defining target ratio and empty dictionary for iteration check
        target_ratio = 0.48
        checked = {}
        # Iteration over potential limit values from 5 to 95
        for potential_value in range(5, 100, 5):
            # Process frame
            processed_frame = core_eye.bilateral_erode_bin(image_eye, potential_value)
            # Calculate size of binarised image
            size_r = Tuning.size(processed_frame)
            checked[potential_value] = size_r
            # Store the size ratio for the current threshold in the dictionary
        optimal_value, optimal_size = min(
            checked.items(), 
            key=(lambda item: abs(item[1] - target_ratio))
        )
        return optimal_value
    
    # Improving the tuning via binarisation limit
    def improve(self, frame_of_eye, LorR):
        # Optimal point calculation for provided image
        binarisation_point = self.optimal_point(frame_of_eye)
        if LorR == 0:
            self.l_limit.append(binarisation_point)
        elif LorR == 1:
            self.r_limit.append(binarisation_point)
        else:
            raise ValueError("LorR must be 0 for the left eye or 1 for the right eye.")

    # Returns the size of the eye's core used for determining direction of look
    @staticmethod
    def size(eye_image):
        # Check if the input is a valid image
        #if eye_image is None or not isinstance(eye_image, np.ndarray):
        #    raise ValueError("Invalid input: eye_image must be a non-empty numpy ndarray.")
        # Check that the image has at least 10 pixels in both dimensions following cropping
        #if eye_image.shape[0] <= 10 or eye_image.shape[1] <= 10:
        #    raise ValueError("Invalid input: eye_image dimensions are too small after cropping.")
        # Crop the image to remove the border
        cropped_image = eye_image[5:-5, 5:-5]
        # Calculate the height and width of the cropped image
        h, w = cropped_image.shape[:2]
        # Calculate the total number of pixels in the cropped image
        total_pixels = h * w
        # Calculate the number of non-zero (black) pixels in the cropped image
        non_zero_pixels = cv2.countNonZero(cropped_image)
        # Calculate the number of unbinarized (zero/black) pixels
        unbinarized_pixels = total_pixels - non_zero_pixels
        # Calculate and return the ratio of unbinarized pixels to total pixels
        return unbinarized_pixels / total_pixels