import math
import numpy as np
import cv2
from .core_eye import core_eye

class Eye(object):
    """
    Extraction of eye -> pupil from an image
    """

    facial_landmarks_left = [36, 37, 38, 39, 40, 41]
    facial_landmarks_right = [42, 43, 44, 45, 46, 47]

    def __init__(self, initial_img, landmarks, side, tuning):
        self.img = None
        self.initial = None
        self.middle = None
        self.pupil = None
        self.facial_lmarks = None

        self._process(initial_img, landmarks, side, tuning)

    @staticmethod
    def _midway(first_point, second_point):
        constant = 2
        coordx = int((first_point.x + second_point.x) / constant)
        coordy = int((first_point.y + second_point.y) / constant)
        return (coordx, coordy)

    def _separate(self, frame, facial_points, pts):
        area = np.array([(facial_points.part(pt).x, facial_points.part(pt).y) for pt in pts])
        area = area.astype(np.int32)
        self.facial_lmarks = area

        # Applying a mask to get only the eye
        h, w = frame.shape[:2]
        frame_blk = np.zeros((h, w), np.uint8)
        mask = np.full((h, w), 255, np.uint8)
        cv2.fillPoly(mask, [area], (0, 0, 0))
        eye = cv2.bitwise_not(frame_blk, frame.copy(), mask=mask)

        #eye selection
        minimum_x = np.min(area[:, 0]) - 5
        maximum_x = np.max(area[:, 0]) + 5
        minimum_y = np.min(area[:, 1]) - 5
        maximum_y = np.max(area[:, 1]) + 5

        self.img = eye[minimum_y:maximum_y, minimum_x:maximum_x]
        self.initial = (minimum_x, minimum_y)

        h, w = self.img.shape[:2]
        self.middle = (w / 2, h / 2)

    def _eye_shut(self, facial_points, pts):
        """detection if eyes are closed or not"""
        l = (facial_points.part(pts[0]).x, facial_points.part(pts[0]).y)
        r = (facial_points.part(pts[3]).x, facial_points.part(pts[3]).y)
        t = self._midway(facial_points.part(pts[1]), facial_points.part(pts[2]))
        b = self._midway(facial_points.part(pts[5]), facial_points.part(pts[4]))

        w_eye = math.hypot((l[0] - r[0]), (l[1] - r[1]))
        h_eye = math.hypot((t[0] - b[0]), (t[1] - b[1]))

        try:
            ratio = w_eye / h_eye
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _process(self, input_f, facial_lmarks, LorR, tuner):
        """Isolation of eye"""
        if LorR == 0:
            eye_points = self.facial_landmarks_left
        elif LorR == 1:
            eye_points = self.facial_landmarks_right
        else:
            return

        self.eye_shut = self._eye_shut(facial_lmarks, eye_points)
        self._separate(input_f, facial_lmarks, eye_points)

        if not tuner.tuning_finished():
            tuner.improve(self.img, LorR)
        threshold = tuner.eye_threshold(LorR)
        self.pupil = core_eye(self.img, threshold)
