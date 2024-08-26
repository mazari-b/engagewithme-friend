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

class EyeMonitoring(object):
    """Position of eyes & whether opened or closed"""

    def __init__(self):
        self.current_frame = None
        self.l_eye = None
        self.r_eye = None
        self.tuning = Tuning()
        
        # UNITTESTING - CSV File with gaze coordinates
        # File paths
        self.excel_path = 'gaze_coordinates.xlsx'
        self.df = pd.DataFrame(columns=['Timestamp', 'X Coordinate', 'Y Coordinate'])

        # _face_detector is used to detect faces
        self._facial_recognition = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        current_directory = os.path.abspath(os.path.dirname(__file__))
        trained_model = os.path.abspath(os.path.join(current_directory, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(trained_model)

    @property
    def detected_ppls(self):
        """Check that the pupils have been located"""
        try:
            int(self.l_eye.pupil.x)
            int(self.l_eye.pupil.y)
            int(self.r_eye.pupil.x)
            int(self.r_eye.pupil.y)
            return True
        except Exception:
            return False

    def _initialise(self):
        """Detects the face and initialize Eye objects"""
        image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        detected_face = self._facial_recognition(image)

        try:
            facial_lmarks = self._predictor(image, detected_face[0])
            self.l_eye = Eye(image, facial_lmarks, 0, self.tuning)
            self.r_eye = Eye(image, facial_lmarks, 1, self.tuning)

        except IndexError:
            self.l_eye = None
            self.r_eye = None

    def update(self, eye):
        """Refreshes the frame and analyzes it."""
        self.current_frame = eye
        self._initialise()
        self.record_coordinates() #UNITTEST

    def coordinates_l(self):
        """Coords of the left pupil"""
        if self.detected_ppls:
            coordx = self.l_eye.initial[0] + self.l_eye.pupil.x #.origin -> .initial
            coordy = self.l_eye.initial[1] + self.l_eye.pupil.y #.origin -> .initial
            return (coordx, coordy)

    def coordinates_r(self):
        """Coords of right pupil"""
        if self.detected_ppls:
            coordx = self.r_eye.initial[0] + self.r_eye.pupil.x #.origin -> .initial
            coordy = self.r_eye.initial[1] + self.r_eye.pupil.y
            return (coordx, coordy)

    def x_plane_direction(self):
        """X plane ratio of where the user is looking: The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.detected_ppls:
            left_p = self.l_eye.pupil.x / (self.l_eye.middle[0] * 2 - 10) #.center -> .middle x4 below
            right_p = self.r_eye.pupil.x / (self.r_eye.middle[0] * 2 - 10)
            return (left_p + right_p) / 2

    def y_plane_direction(self):
        """Y plane ratio of where the user is looking: The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.detected_ppls:
            left_p = self.l_eye.pupil.y / (self.l_eye.middle[1] * 2 - 10)
            right_p = self.r_eye.pupil.y / (self.r_eye.middle[1] * 2 - 10)
            return (left_p + right_p) / 2

    def looking_to_right(self):
        """Returns true if the user is looking to the right"""
        if self.detected_ppls:
            return self.x_plane_direction() <= right_limit

    def looking_to_left(self):
        """Returns true if the user is looking to the left"""
        if self.detected_ppls:
            return self.x_plane_direction() >= left_limit

    def looking_at_centre(self):
        """Using two previous functions to determine if user is making eye contact"""
        if self.detected_ppls:
            return self.looking_to_right() is not True and self.looking_to_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.detected_ppls:
            eyes_closed_r = (self.l_eye.blinking + self.r_eye.blinking) / 2
            return eyes_closed_r > blinking_constant

    def pupils_indicator(self):
        """Graphical indication of the eyes"""
        image = self.current_frame.copy()

        if self.detected_ppls:
            color = (0, 255, 0)
            lx, ly = self.coordinates_l()
            rx, ry = self.coordinates_r()
            cv2.line(image, (lx - 5, ly), (lx + 5, ly), color)
            cv2.line(image, (lx, ly - 5), (lx, ly + 5), color)
            cv2.line(image, (rx - 5, ry), (rx + 5, ry), color)
            cv2.line(image, (rx, ry - 5), (rx, ry + 5), color)

        return image

    def record_coordinates(self):
        """Record the coordinates to an Excel file"""
        if self.detected_ppls:
            timestamp = datetime.now()
            x, y = self.coordinates_l()  # Example: Use left pupil coordinates
            new_data = pd.DataFrame({'Timestamp': [timestamp], 'X Coordinate': [x], 'Y Coordinate': [y]})
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            
            # Save to Excel file
            self.df.to_excel(self.excel_path, index=False)