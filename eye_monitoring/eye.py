import math
import numpy as np
import cv2
from .core_eye import core_eye

# Vision processing from facial to webcam
class Eye(object):
    def __init__(self, image, f_locations, orientation, tuning):
        self.pupil = None
        self.facial_locations= None
        self.img = None
        self.initial = None
        self.end = None
        self.middle = None
        self.process(image, f_locations, orientation, tuning)
        
    # Isolation of facial region
    def separate(self, frame, facial_points, pts):
        # Creation of an array for location marks using specified indices
        area = np.array([(facial_points.part(pt).x, facial_points.part(pt).y) for pt in pts], dtype=np.int32)
        self.facial_locations = area
        # Preparation of a mask for isolation of eye region
        h, w = frame.shape[:2]
        cv2.fillPoly(np.full((h, w), 255, np.uint8), [area], (0, 0, 0))
        region = cv2.bitwise_not(np.zeros((h, w), np.uint8), frame.copy(), mask=np.zeros((h, w), np.uint8))
        # Bounding box of region with minimal margin
        y_lowerlim, y_upperlim = np.min(area[:, 1]) - 5, np.max(area[:, 1]) + 5
        x_lowerlim, x_upperlim = np.min(area[:, 0]) - 5, np.max(area[:, 0]) + 5
        # Extraction of isolated eye region
        self.img = region[y_lowerlim:y_upperlim, x_lowerlim:x_upperlim]
        self.initial = (x_lowerlim, y_lowerlim)
        h, w = self.img.shape[:2]
        h_result, w_result = h/2, w/2
        self.middle = w_result, h_result

    # Calculation of the halfway point of a coordinate pair
    @staticmethod
    def halfpoint(first_point, second_point):
        constant = 2
        cox_result = first_point.x + second_point.x
        coy_result = first_point.y + second_point.y
        coordx = cox_result / constant
        coordy = coy_result / constant
        return coordx, coordy
    
    # Processing of input frame to find region of eye
    def process(self, input_f, facial_locations, LorR, tuner):
        if LorR not in (0, 1):
            return
        eye_points = [36, 37, 38, 39, 40, 41] if LorR == 0 else [42, 43, 44, 45, 46, 47]
        self.eye_shut = self.eye_aspect_r(facial_locations, eye_points)
        self.separate(input_f, facial_locations, eye_points)
        if not tuner.tuning_finished():
            tuner.improve(self.img, LorR)
        threshold = tuner.average_th(LorR)
        self.pupil = core_eye(self.img, threshold)

    # Determines if eyes are closed based on EAR
    # Documentation: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
    def eye_aspect_r(self, facial_points, pts):
        point_l = (facial_points.part(pts[0]).x, facial_points.part(pts[0]).y)
        point_r = (facial_points.part(pts[3]).x, facial_points.part(pts[3]).y)
        point_t = self.halfpoint(facial_points.part(pts[1]), facial_points.part(pts[2]))
        point_b = self.halfpoint(facial_points.part(pts[5]), facial_points.part(pts[4]))
        horizontal = math.hypot((point_l[0] - point_r[0]), (point_l[1] - point_r[1]))
        vertical = math.hypot((point_t[0] - point_b[0]), (point_t[1] - point_b[1]))
        try:
            ear = horizontal / vertical
        except ZeroDivisionError:
            ear = None
        return ear