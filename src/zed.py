from re import S
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

from mebow import pedestrian_orientation

from deeplabv3plus import network
from deeplabv3plus import utils
from deeplabv3plus.datasets import Cityscapes

from PIL import Image
from torchvision import transforms as T
import math
import pyzed.sl as sl

import cv_viewer.tracking_viewer as cv_viewer
import ogl_viewer.viewer as gl
import filter.position_filter as position_filter
import filter.orientation_filter as orientation_filter


import cProfile

from numba import jit, njit

from lib import helpers

import argparse

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt


torch.backends.cudnn.enabled = False

USE_WORLD_COORDINATES = False
PLOT = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', help='svo file path')
    args = parser.parse_args()

    # Read opts
    opts = Opts()

    # Prepare
    color = np.random.randint(0, 255, (100000, 3))  # Create some random colors

    # Init tracker
    tracker = Tracker(PLOT, opts.max_age, opts.max_dist_iou, opts.max_dist_feature, opts.n_init, opts.reinit_interval, opts.point_termi, opts.start_ind,
                    opts.max_size, opts.thre_conf, opts.thre_var_ratio, opts.thre_homo, opts.point_detect, opts.focus_point_manual, opts.focus_point_auto, opts.use_mask, opts.head_detect, opts.
                    r_ratio, opts.interval_num, opts.K, opts.max_point_num, opts.shi_tomasi, opts.feature_params, opts.lk_params)

    # init mebow
    print("initializing MEBOW with resnet18")
    orientation_model = pedestrian_orientation.init_pedestrian_orientation_model()
    print("initializing deeplabV3 with mobileNet")
    num_classes = 19
    decode_fn = Cityscapes.decode_target
    segmentation_model = network.deeplabv3plus_mobilenet(num_classes=num_classes, output_stride=16)
    utils.set_bn_momentum(segmentation_model.backbone, momentum=0.01)
    checkpoint = torch.load("../models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))
    segmentation_model.load_state_dict(checkpoint["model_state"])
    segmentation_model.to(torch.device('cuda'))
    segmentation_model = segmentation_model.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    # frame_ind = opts.start_ind
    frame_ind = 1

    # init zed
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # VGA HD720 HD1080 HD2K
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # PERFORMANCE  QUALITY ULTRA NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 40
    init_params.depth_minimum_distance = 1



    # use camera or svo file if passed as argument
    if (args.svo):
        filepath = args.svo
        print(filepath)
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = False
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    if (args.svo):
        nb_frames = zed.get_svo_number_of_frames()

    quit_app = False
    show2Dviewer = True
    show3Dviewer = False
    show_tracking_viewer = True


    curr_time = 0.0

    cam_pose = sl.Pose()
    py_translation = sl.Translation()

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
    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        

    # store prev frame
    prev_frame = None


    bad_frames = []

    if USE_WORLD_COORDINATES:   
        tracking_params = sl.PositionalTrackingParameters() 
        zed.enable_positional_tracking(tracking_params)
        runtime_parameters = sl.RuntimeParameters(measure3D_reference_frame=sl.REFERENCE_FRAME.WORLD)
    else:
        runtime_parameters = sl.RuntimeParameters()


        
    cam_pos_values_x = []
    cam_pos_values_y = []
    cam_pos_values_z = []

    while (quit_app == False):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            current_fps = zed.get_current_fps()
            # ! skip bad frames
            if zed.get_svo_position() in bad_frames:
                continue

            if (args.svo):
                svo_position = zed.get_svo_position()

            # ! FRAME BY FRAME
            # cv2.waitKey(0) 


            # zed.get_position(cam_c_pose, sl.REFERENCE_FRAME.CAMERA)
            if USE_WORLD_COORDINATES:
                tracking_state = zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
            else:
                tracking_state = zed.get_position(cam_pose, sl.REFERENCE_FRAME.CAMERA)
            # zed.get_position(cam_c_pose, sl.REFERENCE_FRAME.WORLD)

            translation = cam_pose.get_translation(py_translation)

            cam_pos_values_x.append(translation.get()[0])
            cam_pos_values_y.append(translation.get()[1])
            cam_pos_values_z.append(translation.get()[2])

            # text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))

            # frame 1920x1080
            zed.retrieve_image(full_image, sl.VIEW.LEFT, sl.MEM.CPU, full_resolution)
            full_image_left = full_image.get_data()
            frame_full = helpers.zed_img_to_bgr(full_image_left)
            image_render_left = cv2.resize(full_image_left, (960, 540))

            image_snap = cv2.resize(full_image_left, (96, 54))
            # image_render_left = cv2.resize(full_image_left, (1920, 1080))
            # frame 960x540
            # frame = cv2.resize(frame_full, (1920, 1080))
            frame = cv2.resize(frame_full, (960, 540))
            # point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, display_resolution)
            xyz = point_cloud.get_data()
            xyz = helpers.zed_point_cloud_to_xyz(xyz)
            
            prev_time = curr_time
            curr_time = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            dt = (float(curr_time) - float(prev_time)) / 1000000.0
            
            # * segmentation (35ms)
            frame_segmentation_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                img_tensor = transform(frame_segmentation_rgb).unsqueeze(0)
                img_tensor = img_tensor.to(torch.device('cuda'))
            labels = segmentation_model(img_tensor).max(1)[1].cpu().numpy()[0].astype(np.uint8)  # HW
            

            if opts.show_output_images:
                colorized_preds = decode_fn(labels).astype('uint8')
                colorized_preds = cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR)

            # Update tracker  # * (70ms with network, 5ms with tracker)
            tracker.update_frame(frame_ind, frame)
            tracker.update_status_every()
            tracker.update()

            # * Pedestrian orientation
            ped_tensor_list = []
            ped_id_list = []

            for track in tracker.tracks:
                bbox = [int(i) for i in track.bbox]
                if bbox[0] >= 0 and bbox[1] >= 0: # ? DO WE NEED IT?

                    w = bbox[2] * 2
                    h = bbox[3] * 2

                    extrapad = max(w,h) // 10
                    top = max(bbox[1] * 2 - extrapad,0)
                    bottom = min((bbox[1]+bbox[3]) * 2 + extrapad, full_resolution.height - 1)
                    left = max(bbox[0] * 2 - extrapad,0)
                    right = min((bbox[0]+bbox[2]) * 2 + extrapad, full_resolution.width - 1)

                    ped_im = frame_full[ top : bottom, left : right ]


                    head_img = frame_full[ top : top + h // 3, left + w //4 : right - w // 4 ]

                    # cv2.imshow("head", head_img)
                    # cv2.waitKey(0)


                    if ped_im is not None:
                        ped_im = cv2.cvtColor(ped_im, cv2.COLOR_BGR2RGB)

                    # use the full resolution image to get pedestrian images
                    tensor = pedestrian_orientation.convert_to_tensor(Image.fromarray(ped_im))
                    # collect images to array
                    ped_tensor_list.append(tensor)
                    ped_id_list.append(track.track_id)
            # get values from stacked tensor
            orientation_values = pedestrian_orientation.get_prediction(orientation_model, ped_tensor_list)
            # update orientation filters
            for track in tracker.tracks:
                for idx, id in enumerate(ped_id_list):
                    if id == track.track_id:
                        track.update_ori_filter(orientation_values[idx], dt)

            # * 3D position of pedestrians  (about 1ms)
            for track in tracker.tracks:
                # get location of pedestrian
                top = int(track.bbox[1])
                left = int(track.bbox[0])
                bottom = int(track.bbox[1] + track.bbox[3])
                right = int(track.bbox[0] + track.bbox[2])
                # edge cases if bounding box is larger than the image
                if top < 0: 
                    top = 0
                    bottom = int(0 + track.bbox[3])
                if left < 0: 
                    left = 0
                    right = int(0 + track.bbox[2])


                crop_img = labels[top:bottom, left:right]


                height = crop_img.shape[0]
                width = crop_img.shape[1]

                mask = (crop_img == 11)  # pedestrians
                mask_img = mask.astype(np.uint8)  # convert to an unsigned byte
                mask_img *= 255  # in case we want to visualize

                # get a rectangle below the bounding box to determinet what the pedestrian is standing on
                # height of this rectangle is BBwidth / 2
                standing_on = labels[bottom - math.floor(width * 0.5):bottom, left:right]
                road_mask = (standing_on == 0)  # road
                sidewalk_mask = (standing_on == 1)  # sidewalk
                vegetation_mask = ((standing_on == 8) | (standing_on == 9))  # vegetation or terrain
                road_num = np.sum(road_mask)
                sidewalk_num = np.sum(sidewalk_mask)
                vegetation_num = np.sum(vegetation_mask)
                pedestrian_location = "unknown"
                if road_num > 0:
                    if road_num > sidewalk_num + vegetation_num:
                        # on road
                        pedestrian_location = "road"
                    else:
                        pedestrian_location = "edge"
                else:
                    pedestrian_location = "safe"

                x_median, y_median, z_median = helpers.get_3d_position_from_xyz(xyz,top,bottom,left,right)



                # cv2.imshow("frame",frame[top:bottom, left:right])
                # cv2.waitKey(0)
                # print(f"pos_median: {track.x:.2f}, {track.y:.2f}, {track.z:.2f}")

                # ! IS THIS A BUG OR WHAT?
                # ! FIRST POINT CLOUD Z VALUES ARE SEEMINGLY INVERTED EVERY TIME
                if USE_WORLD_COORDINATES:   
                    if zed.get_svo_position() == 1:
                        z_median *= -1 # invert z

                print(f"FRAME {zed.get_svo_position()}")
                # update location of the pedestrian
                track.location = pedestrian_location
                # update 3D position
                track.update_pos_filter(x_median, y_median, z_median, dt)
                # calculate danger value
                track.calculate_danger()



                if PLOT:
                    # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                    track.save_distances(svo_position)


                # print(f"speed: {track.v_x:.2f}, {track.v_y:.2f}, {track.v_z:.2f}, {math.hypot(track.v_x,track.v_y,track.v_z):.2f}")

            # Draw and write
            for track in tracker.tracks:

                if opts.show_output_images:
                    track_id = track.track_id
                    bbox = [int(i) for i in track.bbox]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color[track_id].tolist(), thickness=2)
                    cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[track_id].tolist(), thickness=2)
                    for point in track.points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, color[track_id].tolist(), -1)

            # Show
            if opts.show_output_images:
                height, width = frame.shape[:2]
                cv2.imshow("tracking", frame)
                # cv2.imshow("colorized_preds", colorized_preds)


            # Next frame
            frame_ind += 1

            # Tracking view
            if show_tracking_viewer:
                track_view_generator.generate_view(tracker.tracks, cam_pose, image_track_ocv)
                # track_view_generator.generate_view(tracked_pedestrians, cam_c_pose, image_track_ocv)
            # 2D rendering
            if show2Dviewer:
                np.copyto(image_left_ocv, image_render_left)
                cv_viewer.render_2D(image_left_ocv, image_scale, tracker.tracks, show_bodies=False)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                cv2.imshow("2D View and Birds View", global_image)
                cv2.waitKey(1)


            if (args.svo):
                if svo_position >= (nb_frames - 1):  # End of SVO
                # if svo_position >= 20:  # ! TEMP FIX
                    print("end of svo")

                    if PLOT:
                        for track in tracker.old_tracks:
                            plt.plot(track.frames, track.collision_distances, label = "Collision distance")
                            # # naming the x axis
                            plt.xlabel('frame')
                            # # naming the y axis
                            plt.ylabel('dist (m)')
                            # # giving a title to my graph
                            # plt.title(f'Track {track.track_id}')
                            # # show a legend on the plot
                            # plt.legend()

                            # plt.plot( track.x_values, track.z_values, label = "Trajectory")
                            plt.show()

                            # EXPORT_CSV
                            f = open(f'../track_{track.track_id}.csv', 'w')
                            f.write("frame,dt,d\n")
                            for line in zip(track.frames, track.times, track.collision_distances):
                                f.write(f"{line[0]},{line[1]},{line[2]}\n")
                            f.close()

                            ax = plt.axes(projection='3d')
                            ax.set_box_aspect((np.ptp(cam_pos_values_x + track.x_values), 
                                                np.ptp(cam_pos_values_y + track.y_values), 
                                                np.ptp(cam_pos_values_z + track.z_values)))
                            ax.plot3D(cam_pos_values_x, cam_pos_values_x, cam_pos_values_z, 'gray')
                            ax.plot3D(track.x_values, track.y_values, track.z_values,  'red')

                            plt.show()

            if cv2.waitKey(1) == 27:
                quit_app = True
                print("quitting")



if __name__ == "__main__":
    main()