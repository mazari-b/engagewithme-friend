from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .tuning import Tuning
import pandas as pd
from datetime import datetime

#Constants
right_limit = 0.35
left_limit = 0.65
blinking_constant = 0.38 * 10
l_constant = 0
r_constant = 1

# Determines the state of eyes and what actions
class EyeMonitoring(object):
    def __init__(self):
        self.current_image = None
        self.l_eye = None
        self.r_eye = None
        self.tuning = Tuning()
        
        # UNITTESTING - CSV File with gaze coordinates
        # File paths
        #self.excel_path = 'gaze_coordinates.xlsx'
        #self.df = pd.DataFrame(columns=['Timestamp', 'X Coordinate', 'Y Coordinate'])

        # used to detect a face using default Dlib
        self._profile_recognition = dlib.get_frontal_face_detector()
        current_directory = os.path.abspath(os.path.dirname(__file__))
        trained_model = os.path.abspath(os.path.join(current_directory, "ml_face_models/shape_predictor_68_face_landmarks.dat"))
        self._sp = dlib.shape_predictor(trained_model)
        
    # Set eye & facial marks
    def initialise(self):
        # Selects current image from webcam and recognises a profile (face)
        current_image = self.current_image
        image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        detected_profile = self._profile_recognition(image)
        try:
            # attempt to land points around eyes
            detected_profile_first_element = detected_profile[0]
            facial_lmarks = self._sp(image, detected_profile[l_constant])
            self.l_eye = Eye(image, facial_lmarks, l_constant, self.tuning)
            self.r_eye = Eye(image, facial_lmarks, r_constant, self.tuning)
        except:
            # assume eyes do not exist
            EmptyState = None
            self.l_eye = EmptyState
            self.r_eye = EmptyState
        
    # Updates the image the algorithm is working with
    def update(self, image):
        current_image = image
        self.current_image = current_image
        self.initialise()
        #self.record_coordinates() #UNITTEST    

    # Determines the state of eyes and what actions
    @property
    def detected_ppls(self):
        try:
            # Attempt to convert coordinates to integers
            l_pupil_x = int(self.l_eye.pupil.x)
            l_pupil_y = int(self.l_eye.pupil.y)
            r_pupil_x = int(self.r_eye.pupil.x)
            r_pupil_y = int(self.r_eye.pupil.y)
            return True
        except Exception:
            return False
        
    # The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0
    def x_plane_direction(self):
        if not self.detected_ppls:
            return None
        left_p = self.l_eye.pupil.x / (self.l_eye.middle[0] * 2 - 10) #.center -> .middle x4 below
        right_p = self.r_eye.pupil.x / (self.r_eye.middle[0] * 2 - 10)
        total_p = left_p + right_p
        result = total_p/2
        return result

    # The extreme top is 0.0, the center is 0.5 and the extreme bottom is 1.0 
    def y_plane_direction(self):
        if not self.detected_ppls:
            return None
        left_p = self.l_eye.pupil.y / (self.l_eye.middle[1] * 2 - 10)
        right_p = self.r_eye.pupil.y / (self.r_eye.middle[1] * 2 - 10)
        total_p = left_p + right_p
        result = total_p/2
        return result

    # Coords of left core
    def coordinates_l(self):
        if not self.detected_ppls:
            return None
        coordx = self.l_eye.initial[l_constant] + self.l_eye.pupil.x #.origin -> .initial
        coordy = self.l_eye.initial[1] + self.l_eye.pupil.y #.origin -> .initial
        return (coordx, coordy)

    # Coords of right core
    def coordinates_r(self):
        if not self.detected_ppls:
            return None
        coordx = self.r_eye.initial[0] + self.r_eye.pupil.x #.origin -> .initial
        coordy = self.r_eye.initial[r_constant] + self.r_eye.pupil.y
        return (coordx, coordy)

    # Returns boolean: True if Closed, False if Other
    def is_blinking(self):
        if self.detected_ppls:
            sum_blinking = self.l_eye.blinking + self.r_eye.blinking
            eyes_closed_r = sum_blinking * 0.5
            return eyes_closed_r > blinking_constant

    # Returns boolean: True if Right, False if Other
    def looking_to_right(self):
        if not self.detected_ppls:
            return False
        # Calculate and return whether the x-plane direction is to the right
        return self.x_plane_direction() <= right_limit

    # Returns boolean: True if Left, False if Other
    def looking_to_left(self):
        if not self.detected_ppls:
            return False
        # Calculate and return whether the x-plane direction is to the left
        return self.x_plane_direction() >= left_limit

    # Returns boolean: True if Centre, False if Other
    def looking_at_centre(self):
        if not self.detected_ppls:
            return False
        # Determine if  looking not to the right or left, implying it is centered
        return not self.looking_to_right() and not self.looking_to_left()

    # Recording coordinates for testing purpose
    def record_coordinates(self):
        """Record the coordinates to an Excel file"""
        if self.detected_ppls:
            timestamp = datetime.now()
            x, y = self.coordinates_l()  # Example: Use left pupil coordinates
            new_data = pd.DataFrame({'Timestamp': [timestamp], 'X Coordinate': [x], 'Y Coordinate': [y]})
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            
            # Save to Excel file
            self.df.to_excel(self.excel_path, index=False)