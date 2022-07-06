import cv2
import numpy as np

# http://campar.in.tum.de/Chair/KalmanFilter
"""
This is a Kalman filter implementation for pedestrian 3d position

"""

"""
dynamparams = 9
measureparams = 3

statematrix: 
x_k=[x,y,z,x',y',z',x'',y'',z'']T

transition matrix:
Note: set dT at each processing step!
http://campar.in.tum.de/Chair/KalmanFilter
[
	1		0 	0 	dt 	0 	0 	dt^2/2	0				0
	0		1		0		0		dt	0		0				dt^2/2	0
	0		0		1		0		0		dt	0				0				dt^2/2
	0		0		0		1		0		0		dt			0				0
	0		0		0		0		1		0		0				dt			0	
	0		0		0		0		0		1		0				0				dt
	0		0		0		0		0		0		1				0				0
	0		0		0		0		0		0		0				1				0
	0		0		0		0		0		0		0				0				1	
]

measurement matrix:
[
	1	0	0	0	0	0	0	0	0
	0	1	0	0	0	0	0	0	0
	0	0	1	0	0	0	0	0	0
]

"""

class PositionFilter:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        # dynamic parameters: 9, measurement paramateres: 3
        self.kf = cv2.KalmanFilter(9, 3, 0)
        self.kf.measurementMatrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                              [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                              [0., 0., 1., 0., 0., 0., 0., 0., 0.]], np.float32)
        self.kf.statePre = np.array([self.x, self.y, self.z, 0., 0., 0., 0., 0., 0.], np.float32)
        self.kf.statePost = np.array([self.x, self.y, self.z, 0., 0., 0., 0., 0., 0.], np.float32)
        # Q: process noise covariance
        self.kf.processNoiseCov = cv2.setIdentity(self.kf.processNoiseCov, 0.0001)
        #  R: measurement noise covariance
        self.kf.measurementNoiseCov = cv2.setIdentity(self.kf.measurementNoiseCov, 0.1)
        # self.kf.errorCovPost = cv2.setIdentity(self.kf.errorCovPost, 1e-4)
        #  Q, the process noise covariance, contributes to the overall uncertainty.
        #  When Q is large, the Kalman Filter tracks large changes in the data more closely than for smaller Q.
        #  R, the measurement noise covariance, determines how much information from the measurement is used.
        #  If R is high, the Kalman Filter considers the measurements as not very accurate. For smaller R it will follow the measurements more closely.

    def predict(self, dt):
        self.kf.transitionMatrix = np.array([[1.,		0., 	0., 	dt, 	0., 	0., 	dt*dt*0.5,	0.,				  0.],
                                             [0.,		1., 	0., 	0, 	    dt, 	0., 	0.,	        dt*dt*0.5,        0.],
                                             [0.,		0., 	1., 	0, 	    0., 	dt, 	0.,	        0.,				  dt*dt*0.5],
                                             [0.,		0., 	0., 	1., 	0., 	0., 	dt,	        0.,				  0.],
                                             [0.,		0., 	0., 	0., 	1., 	0., 	0.,	        dt,				  0.],
                                             [0.,		0., 	0., 	0., 	0., 	1., 	0.,	        0.,				  dt],
                                             [0.,		0., 	0., 	0., 	0., 	0., 	1.,	        0.,				  0.],
                                             [0.,		0., 	0., 	0., 	0., 	0., 	0.,	        1.,				  0.],
                                             [0.,		0., 	0., 	0., 	0., 	0., 	0.,	        0.,				  1.]], np.float32)
        prediction = self.kf.predict()
        self.x = prediction[0]
        self.y = prediction[1]
        self.z = prediction[2]
        self.v_x = prediction[3]
        self.v_y = prediction[4]
        self.v_z = prediction[5]

    def correct(self, measured_x, measured_y, measured_z):
        corrected = self.kf.correct(np.array([measured_x, measured_y, measured_z], np.float32))
        self.x = corrected[0]
        self.y = corrected[1]
        self.z = corrected[2]
        self.v_x = corrected[3]
        self.v_y = corrected[4]
        self.v_z = corrected[5]
