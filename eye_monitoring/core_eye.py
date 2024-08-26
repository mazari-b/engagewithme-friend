import numpy as np
import cv2

class core_eye(object):
    """
    Narrows down to iris and calculates gaze direction
    """
    preprocess_image_counter = 1
    binary_image_counter = 1
    core_image_counter = 1

    def __init__(self, image_eye, binarisation_point):
        self.image_iris = None
        self.binarisation_point = binarisation_point
        self.x = None
        self.y = None

        self.locate_core_of_eye(image_eye)

    @staticmethod
    def preprocess_image(image_eye, threshold):
        """Performs operations on the eye frame to prepare to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        filtered_image = cv2.bilateralFilter(image_eye, 10, 15, 15)
        eroded_image = cv2.erode(filtered_image, kernel, iterations=3)
        binary_image = cv2.threshold(eroded_image, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # UNIT TESTING
        #cv2.imwrite(f"preprocessed_image{core_eye.preprocess_image_counter}.png", filtered_image)
        #cv2.imwrite(f"binary_image{core_eye.binary_image_counter}.png", binary_image)
        
        # UNIT TESTING - Increment the counters for the next call
        #core_eye.preprocess_image_counter += 1
        #core_eye.binary_image_counter += 1

        return binary_image

    def locate_core_of_eye(self, image_eye):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.image_iris = self.preprocess_image(image_eye, self.binarisation_point)

        contours, _ = cv2.findContours(self.image_iris, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        ctrs_list = sorted(contours, key=cv2.contourArea)

        #contour_image = np.copy(image_eye)

        try:
            value_mmts = cv2.moments(ctrs_list[-2])
            self.x = int(value_mmts['m10'] / value_mmts['m00'])
            self.y = int(value_mmts['m01'] / value_mmts['m00'])
            
            # Draw the contours and centroid on the image
            #cv2.drawContours(contour_image, [ctrs_list[-2]], -1, (0, 255, 0), 2)
            #cv2.circle(contour_image, (self.x, self.y), 5, (0, 0, 255), -1)
        except (IndexError, ZeroDivisionError):
            pass
        #cv2.imwrite(f"core_image_{core_eye.core_image_counter}.png", contour_image)
        #core_eye.core_image_counter += 1
