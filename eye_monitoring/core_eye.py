import numpy as np
import cv2
outline_global_index = -2
outlines_index = -2
spatial_moment_m00, spatial_moment_m01, spatial_moment_m10 = 'm00', 'm01', 'm10' 

# Initialising class to detect core
class core_eye(object):
    preprocess_image_counter = 1
    binary_image_counter = 1
    core_image_counter = 1

    def __init__(self, image, binarisation_point):
        self.x = None
        self.y = None
        self.coreimage = None
        self.binarisation_point = binarisation_point
        self.locate_core_of_eye(image)
        
    def locate_core_of_eye(self, image):
        # Location of the core of eye using filtering and modification methods
        # Documentation: https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        self.coreimage = self.bilateral_erode_bin(image, self.binarisation_point)
        outlines, _ = cv2.findContours(self.coreimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[outline_global_index:]
        outlines_list = sorted(outlines, key=cv2.contourArea)
        try:
            value_mmts = cv2.moments(outlines_list[outline_global_index])
            self.x = int(value_mmts[spatial_moment_m10] / value_mmts[spatial_moment_m00])
            self.y = int(value_mmts[spatial_moment_m01] / value_mmts[spatial_moment_m00])
            # Draw the contours and centroid on the image
            #cv2.drawContours(contour_image, [outlines_list[-2]], -1, (0, 255, 0), 2)
            #cv2.circle(contour_image, (self.horizontalplane, self.verticalplane), 5, (0, 0, 255), -1)
        except:
            pass
        #cv2.imwrite(f"core_image_{core_eye.core_image_counter}.png", contour_image)
        #core_eye.core_image_counter += 1

    @staticmethod
    def bilateral_erode_bin(image, threshold):
        # Filtering and applications on image
        # Documentation: https://medium.com/@nimritakoul01/image-processing-using-opencv-python-9c9b83f4b1ca
        convolution_matrix = np.ones((3, 3))
        filtered_image = cv2.bilateralFilter(image, 11, 16, 16)
        eroded_image = cv2.erode(filtered_image, convolution_matrix, 3)
        binary_image = cv2.threshold(eroded_image, threshold, 255, cv2.THRESH_BINARY)
        binary_image = binary_image[1]
        
        # UNIT TESTING
        #cv2.imwrite(f"preprocessed_image{core_eye.preprocess_image_counter}.png", filtered_image)
        #cv2.imwrite(f"binary_image{core_eye.binary_image_counter}.png", binary_image)
        
        # UNIT TESTING - Increment the counters for the next call
        #core_eye.preprocess_image_counter += 1
        #core_eye.binary_image_counter += 1
        return binary_image