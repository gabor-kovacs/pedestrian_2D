import cv2
import numpy as np

# http://campar.in.tum.de/Chair/KalmanFilter
"""
This is a Kalman filter implementation for rotation
To solve issues with wrapping at 0deg-360deg, total rotations are tracked

"""


class OrientationFilter:
    def __init__(self, orientation):
        self.rotation_value = orientation * np.pi / 180.0
        self.orientation = orientation
        self.kf = cv2.KalmanFilter(3, 1, 0)
        self.kf.measurementMatrix = np.array([[1., 0., 0.]], np.float32)
        # Q: process noise covariance
        self.kf.processNoiseCov = cv2.setIdentity(self.kf.processNoiseCov, 0.1)
        #  R: measurement noise covariance
        self.kf.measurementNoiseCov = cv2.setIdentity(self.kf.measurementNoiseCov, 0.1)
        # self.kf.errorCovPost = cv2.setIdentity(self.kf.errorCovPost, 1e-4)
        #  Q, the process noise covariance, contributes to the overall uncertainty.
        #  When Q is large, the Kalman Filter tracks large changes in the data more closely than for smaller Q.
        #  R, the measurement noise covariance, determines how much information from the measurement is used.
        #  If R is high, the Kalman Filter considers the measurements as not very accurate. For smaller R it will follow the measurements more closely.

    def predict(self, dt):
        self.kf.transitionMatrix = np.array(
            [[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]], np.float32
        )
        prediction = self.kf.predict()
        self.rotation_value = prediction[0][0]

        temp_rotation_value = self.rotation_value
        if temp_rotation_value > 0:
            while temp_rotation_value > (2 * np.pi):
                temp_rotation_value -= 2 * np.pi
        if temp_rotation_value < 0:
            while temp_rotation_value < 0:
                temp_rotation_value += 2 * np.pi
        self.orientation = int(temp_rotation_value * 180 / np.pi)

    def correct(self, orientation):
        measured_orientation_rad = orientation * np.pi / 180.0
        temp_rotation_value = self.rotation_value

        if temp_rotation_value > 0:
            while temp_rotation_value > (2 * np.pi):
                temp_rotation_value -= 2 * np.pi
        if temp_rotation_value < 0:
            while temp_rotation_value < 0:
                temp_rotation_value += 2 * np.pi

        rotation_diff = measured_orientation_rad - temp_rotation_value
        # * this is between -2PI and 2PI, but if it is more than PI, it is closer from the other direction
        if rotation_diff < -1 * np.pi:
            rotation_diff += (2 * np.pi)
        if rotation_diff > np.pi:
            rotation_diff -= (2 * np.pi)

        measurement = self.rotation_value + rotation_diff
        corrected = self.kf.correct(np.array([measurement], np.float32))
        self.rotation_value = corrected[0][0]
        # convert to degrees to draw in opencv
        temp_rotation_value = self.rotation_value

        if temp_rotation_value > 0:
            while temp_rotation_value > (2 * np.pi):
                temp_rotation_value -= 2 * np.pi

        if temp_rotation_value < 0:
            while temp_rotation_value < 0:
                temp_rotation_value += 2 * np.pi

        self.orientation = int(temp_rotation_value * 180 / np.pi)
