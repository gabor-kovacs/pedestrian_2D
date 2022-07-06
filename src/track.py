import numpy as np
import statistics
from pyparsing import null_debug_action
import scipy.stats as stats
from scipy.spatial import distance
import pandas as pd

import math
from lib.pos_filter import PositionFilter
from lib.ori_filter import OrientationFilter

def calc_mu_cov(x):
    mu = x.mean(axis=0)
    cov = np.cov(x.T)

    return mu, cov


# Hotelling method
def denoise(x):
    mu, cov = calc_mu_cov(x)
    cov_inv = np.linalg.pinv(cov)
    dist = np.array([distance.mahalanobis(i, mu, cov_inv)**2 for i in x])
    thre =stats.chi2.isf(0.1, 2)
    x_denoise = x[dist < thre]

    return x_denoise


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, track_id, bbox, feature, mask, n_init, max_age, reinit_interval):
        self.track_id = track_id
        self.state = TrackState.Tentative
        self.hits = 1
        self.matches = 1
        self.age = 1
        self.no_match_num = 0
        self.bbox = bbox # x, y, w, h
        self.feature = feature
        self.mask = mask
        self.points = []
        self.center = [self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2]
        self.n_init = n_init
        self.max_age = max_age
        self.reinit_interval = reinit_interval

        self.pos_filter = None # ! init and update when 3D pos is known already
        self.x = None
        self.y = None
        self.z = None
        self.v_x = 0.
        self.v_y = 0.
        self.v_z = 0.
        self.orientation = 180
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[0] + bbox[2]
        self.bottom = bbox[1] + bbox[3]
        self.location = "unknown"
        # store positions in the camera coordinate frame
        self.position = np.array([self.x, self.y, self.z])
        
        self.ori_filter = None # ! init and update when 3D pos is known already
        self.rotation_value = 0.
        self.view_angle_deg = 0
        # danger model
        self.awareness_level = 0
        self.location_level = 0
        self.d = 0  # collision distance
        self.collision_level = 0
        self.safety_level = 0

        self.dt = 0.0
        # ! PLOT 
        self.frames = []
        self.collision_distances = []
        self.distances = []
        self.x_values = []
        self.y_values = []
        self.z_values = []
        self.times = []

        self.awareness_levels = []
        self.location_levels = []
        self.collision_levels = []
        self.safety_levels = []

        # self_speeds_calc = []    
        # self_speeds_filter = []    




    def update_status_every(self):
        self.age += 1

    def update_status_hits(self): # for starting track      
        self.hits += 1
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def update_by_points(self, new_points):
        # Update position (x, y)
        delta = new_points - self.points
        delta_norm = [np.linalg.norm(np.array(i)) for i in delta.tolist()]
        median = statistics.median_low(delta_norm)
        median_ind = delta_norm.index(median)
        self.bbox[0] = self.bbox[0] + delta[median_ind][0]
        self.bbox[1] = self.bbox[1] + delta[median_ind][1]

        self.left = self.bbox[0]
        self.top = self.bbox[1]
        self.right = self.bbox[0] + self.bbox[2]
        self.bottom = self.bbox[1] + self.bbox[3]

        # Update scale (w, h)
#        if len(new_points) >= 2:
#            new_points_denoise = denoise(new_points)
#            _, cov_new = calc_mu_cov(new_points_denoise)
#            cov_new += 0.01 * np.ones([2, 2])
#            old_points_denoise = denoise(old_points)
#            _, cov_old = calc_mu_cov(old_points_denoise)
#            cov_old += 0.01 * np.ones([2, 2])
#            self.bbox[2] = self.bbox[2] * cov_new[0][0] / cov_old[0][0]
#            self.bbox[3] = self.bbox[3] * cov_new[1][1] / cov_old[1][1]

        # Update points
        self.points = new_points

        # Update status
        self.update_status_hits()

    def update_by_det(self, det, feature, mask, already_updated):
        self.bbox = det
        self.feature = feature
        self.mask = mask
        self.no_match_num = 0
        if already_updated == 0:
            self.update_status_hits()

        self.left = self.bbox[0]
        self.top = self.bbox[1]
        self.right = self.bbox[0] + self.bbox[2]
        self.bottom = self.bbox[1] + self.bbox[3]

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.no_match_num += 1

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.no_match_num >= self.max_age / self.reinit_interval:
            self.state = TrackState.Deleted

    def mark_deleted(self):
        """Mark this track as deleted
        """
        self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


    def update_pos_filter(self, x, y, z, dt):
        """Update the 3D position Kalman filter"""
        # if this is the first time, init a new tracker
        if self.pos_filter is None:
            self.pos_filter = PositionFilter(x,y,z)

        # ! DEBUG
        # speed from kalman:
        print(f"speed from kalman: {math.hypot(self.v_x, self.v_y, self.v_z):.2f}")

        prev_x = self.pos_filter.x
        prev_y = self.pos_filter.y
        prev_z = self.pos_filter.z

        # if already tracked, update the values
        self.pos_filter.predict(dt)
        self.pos_filter.correct(x,y,z)

        next_x = self.pos_filter.x
        next_y = self.pos_filter.y
        next_z = self.pos_filter.z

        if prev_x: # skip init
            self.v_x = (next_x - prev_x) / dt
            self.v_y = (next_y - prev_y) / dt
            self.v_z = (next_z - prev_z) / dt
            # print(f"previous position: {self.x, self.y, self.z}")
            # print(f"current position: {x, y, z}")
            # delta = math.hypot(prev_x-next_x, prev_y-next_y, prev_z-next_z)
            # print(f"delta: {delta}")
            # print(f"dt: {dt}")
            # speed = delta / dt
            # print(f"speed from location diff: {speed:.2f}")

        # values stored inside the filter itself are copied to the pedestrian object
        self.x = self.pos_filter.x
        self.y = self.pos_filter.y
        self.z = self.pos_filter.z
        # self.v_x = self.pos_filter.v_x
        # self.v_y = self.pos_filter.v_y
        # self.v_z = self.pos_filter.v_z
        self.position = np.array([self.x, self.y, self.z])
        self.dt = dt




    def update_ori_filter(self, orientation, dt):
        """Update the orientation Kalman filter"""
        # if this is the first time, init a new tracker
        if self.ori_filter is None:
            self.ori_filter = OrientationFilter(orientation)
        # if already tracked, update the values
        self.ori_filter.predict(dt)
        self.ori_filter.correct(orientation)
        # values stored inside the filter itself are copied to the pedestrian object
        self.rotation_value = self.ori_filter.rotation_value
        self.orientation = self.ori_filter.orientation
        self.dt = dt


    def calculate_danger(self):

        alpha = abs(self.orientation - 180.0) 

        # awareness level
        awareness_level = np.max([ ((0.33 -1.0) / 110. * alpha) + 1.0, 0.33])

        # awareness_level = 0
        # if (self.orientation > 70 and self.orientation < 290):
        #     awareness_level = 0.2
        # if (self.orientation > 120 and self.orientation < 240):
        #     awareness_level = 0.5
        # if (self.orientation > 150 and self.orientation < 210):
        #     awareness_level = 0.7
        # if (self.orientation > 170 and self.orientation < 190):
        #     awareness_level = 1
        self.awareness_level = awareness_level

        # location
        location_level = 0.33
        if (self.location == "unknown"):
            location_level = 0.33
        if (self.location == "road"):
            location_level = 0.33
        if (self.location == "edge"):
            location_level = 0.66
        if (self.location == "safe"):
            location_level = 1
        self.location_level = location_level

        # collision
        # 2 points of the current tracjectory are (ped.x, - ped.z) and (ped.x + v_x , - (ped.z + v_z ) )
        # get the distance from the car to thi line defined by this
        # https://en.wikipedia.org/w/index.php?title=Distance_from_a_point_to_a_line&action=edit&section=3
        # d = |( v_x ) * -z - x * ( -v_z )| / sqrt(v_x^2 + v_z^2)
        # d = np.abs((self.v_x * -1 * self.z) - (self.x * -1 * self.v_z)

        # d = abs((self.v_x * -1 * self.z) - (self.x * -1 * self.v_z)
        #             / math.hypot(self.v_x, self.v_z))

        # ! use 3d formula instead of 2d
        tmp = np.cross([self.x, self.y, self.z], [self.v_x, self.v_y, self.v_z])

        if math.hypot(self.v_x, self.v_y, self.v_z) == 0:
            d = 0.0
        else:
            d = math.hypot(tmp[0], tmp[1], tmp[2]) / math.hypot(self.v_x, self.v_y, self.v_z)

        self.d = d



        # print(d)

        collision_level =  (d - 1.5) / 2.5
        if collision_level < 0:
            collision_level = 0.0
        if collision_level > 1:
            collision_level = 1.0

        # collision_level =  np.max([ np.min([(d - 1.0) / 4.0 , 0]), 1]) 

        # collision_level = 0
        # if (d > 1):
        #     collision_level = 0.2
        # if (d > 2):
        #     collision_level = 0.5
        # if (d > 3):
        #     collision_level = 0.7
        # if (d > 5):
        #     collision_level = 1
        # # if the pedestrian is getting away from the car, then this does not apply
        # if (self.v_z < 0):
        #     collision_level = 1
        # # if the velocity is lower than 3 m/s, it is safe
        # if (math.hypot(self.v_x, self.v_z) < 2):
        #     collision_level = 1

        self.collision_level = collision_level

        # self.safety_level = math.hypot(self.v_x,self.v_y,self.v_z)
        self.safety_level = self.collision_level * (self.location_level + self.awareness_level) / 2.0

        # danger level is the suim of these values
        # if (math.hypot(self.v_x,self.v_y,self.v_z) > 2):
        #     self.safety_level = self.collision_level * (self.location_level + self.awareness_level) / 2.0
        # else:
        #     self.safety_level = 1.0

    def save_distances(self, frame):
        self.frames.append(frame)
        self.distances.append(math.hypot(self.x,self.y,self.z))
        self.collision_distances.append(self.d)
        self.x_values.append( self.x )
        self.y_values.append( self.y )
        self.z_values.append( self.z )
        self.times.append( self.dt )

        self.awareness_levels.append(self.awareness_level)
        self.location_levels.append(self.location_level)
        self.collision_levels.append(self.collision_level)
        self.safety_levels.append(self.safety_level)