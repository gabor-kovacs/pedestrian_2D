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
import torch
import numpy as np
sys.path.append("./fairmot/lib")
# sys.path.append("../FairMOT/src/lib")

from mebow import pedestrian_orientation

from deeplabv3plus import network
from deeplabv3plus import utils
from deeplabv3plus.datasets import Cityscapes

from PIL import Image
from torchvision import transforms as T

import pyzed.sl as sl


# * init zed
# Create a Camera object
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # VGA HD720 HD1080 HD2K
init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # PERFORMANCE  QUALITY ULTRA
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
init_params.depth_maximum_distance = 20
init_params.depth_minimum_distance = 1
init_params.svo_real_time_mode = False
init_params.set_from_svo_file("/home/gabor/Documents/pedestrianV3/videos/short.svo")
# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

quit_app = False
zed_image = sl.Mat()
display_resolution = sl.Resolution(960, 540)
while (quit_app == False):
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        svo_position = zed.get_svo_position()
        current_fps = zed.get_current_fps()
        print(f"current fps: {current_fps}")
        # Retrieve left image
        zed.retrieve_image(zed_image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        frame = zed_image.get_data()
        # frame = frame[:, :, 0:3] # remove transparency channel
        # frame = np.copy(frame) # stupid but necessary
        # Show
        height, width = frame.shape[:2]
        # smaller = cv2.resize(frame, (round(width / 4), round(height / 4)), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("tracking", frame)
        # cv2.waitKey(1)
        if cv2.waitKey(1) == 27:
            quit_app = True
# Release
cv2.destroyAllWindows()

