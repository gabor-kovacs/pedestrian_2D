from time import time
from opts import Opts
from tracker import Tracker
from pytz import timezone
import motmetrics as mm
import datetime
import pickle
import sys
import os
import cv2



import numpy as np

import math
import pyzed.sl as sl


import cProfile

from numba import jit, njit

from lib import helpers

import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', help='svo file path')
    from_svo = parser.parse_args()

    # init zed
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # VGA HD720 HD1080 HD2K
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # PERFORMANCE  QUALITY ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 40
    init_params.depth_minimum_distance = 1

    # use camera or svo file if passed as argument
    if (from_svo.svo):
        filepath = from_svo.svo
        print(filepath)
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = False
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    quit_app = False



    cam_c_pose = sl.Pose()
    full_image = sl.Mat()
    point_cloud = sl.Mat()   
    display_resolution = sl.Resolution(960, 540)
    full_resolution = sl.Resolution(1920, 1080)
    image_scale = [1, 1]
    image_left_ocv = np.full(
        (display_resolution.height, display_resolution.width, 4),
        [245, 239, 239, 255],
        np.uint8,
    )




    FRAME = 181

    while (quit_app == False):
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # ! get the timestamp of the frame we want to sync with the lidar data
            if zed.get_svo_position() == FRAME:
                zed.retrieve_image(full_image, sl.VIEW.LEFT, sl.MEM.CPU, full_resolution)
                # camera position
                zed.get_position(cam_c_pose, sl.REFERENCE_FRAME.CAMERA)
                # frame 1920x1080
                full_image_left = full_image.get_data()
                frame_full = helpers.zed_img_to_bgr(full_image_left)
                image_render_left = cv2.resize(full_image_left, (960, 540))

                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                print("***")
                print(timestamp)
                print("***")

                cv2.imshow("im", image_render_left)
                cv2.waitKey(0)


   
            if cv2.waitKey(1) == 27:
                quit_app = True


if __name__ == "__main__":
    main()