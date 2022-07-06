import cv2
import numpy as np

# http://campar.in.tum.de/Chair/KalmanFilter
"""
This is a Kalman filter implementation for pedestrian 3d position

"""

"""
dynamparams = 2
measureparams = 1

statematrix: 
x_k=[x,x']T

transition matrix:
Note: set dT at each processing step!
http://campar.in.tum.de/Chair/KalmanFilter
[1  dt 
 0  1]

measurement matrix:
[1	0]
"""

class OneDimensionalFilter:
    def __init__(self, x):
        self.x = x

        # dynamic parameters: 2, measurement paramateres: 1
        self.kf = cv2.KalmanFilter(2, 1, 0)
        self.kf.measurementMatrix = np.array([[1., 0.]], np.float32)
        self.kf.statePre = np.array([self.x, 0.], np.float32)
        self.kf.statePost = np.array([self.x, 0.], np.float32)
        # Q: process noise covariance
        self.kf.processNoiseCov = cv2.setIdentity(self.kf.processNoiseCov, 0.001)
        #  R: measurement noise covariance
        self.kf.measurementNoiseCov = cv2.setIdentity(self.kf.measurementNoiseCov, 0.1)
        # self.kf.errorCovPost = cv2.setIdentity(self.kf.errorCovPost, 1e-4)
        #  Q, the process noise covariance, contributes to the overall uncertainty.
        #  When Q is large, the Kalman Filter tracks large changes in the data more closely than for smaller Q.
        #  R, the measurement noise covariance, determines how much information from the measurement is used.
        #  If R is high, the Kalman Filter considers the measurements as not very accurate. For smaller R it will follow the measurements more closely.

    def predict(self, dt):
        self.kf.transitionMatrix = np.array([[1.,	dt],
                                             [0.,	1.]], np.float32)
        prediction = self.kf.predict()
        self.x = prediction[0]
        self.v_x = prediction[1]

    def correct(self, measured_x):
        corrected = self.kf.correct(np.array([measured_x], np.float32))
        self.x = corrected[0]
        self.v_x = corrected[1]

    def update(self, x, dt):
        self.predict(dt)
        self.correct(x)
