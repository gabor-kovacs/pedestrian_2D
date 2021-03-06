import cv2
import numpy as np
from torch.nn.modules.container import T

from cv_viewer.utils import *
import pyzed.sl as sl
import math
from collections import deque

import random
import colorsys

# ----------------------------------------------------------------------
#       2D LEFT VIEW
# ----------------------------------------------------------------------

# np.random.seed(1)
# color = np.random.randint(0, 255, (100000, 3))  # Create some random colors

def generate_color_from_id(id):
    random.seed(id)
    h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
    return [b, g, r, 255]
    # return [color[id].tolist()[0], color[id].tolist()[1], color[id].tolist()[2], 255]


def cvt(pt, scale):
    """
    Function that scales point coordinates
    """
    out = [pt[0] * scale[0], pt[1] * scale[1]]
    return out


def get_image_position(bounding_box_image, img_scale):
    out_position = np.zeros(2)
    out_position[0] = (
        bounding_box_image[0][0]
        + (bounding_box_image[2][0] - bounding_box_image[0][0]) * 0.5
    ) * img_scale[0]
    out_position[1] = (
        bounding_box_image[0][1]
        + (bounding_box_image[2][1] - bounding_box_image[0][1]) * 0.5
    ) * img_scale[1]
    return out_position


def render_2D(left_display, img_scale, pedestrians, show_bodies=False):
    overlay = left_display.copy()

    line_thickness = 2
    for pedestrian in pedestrians:
        base_color = generate_color_from_id(pedestrian.track_id)
        # Display image scaled 2D bounding box
        top_left_corner = cvt([pedestrian.left, pedestrian.top], img_scale)
        top_right_corner = cvt([pedestrian.right, pedestrian.top], img_scale)
        bottom_right_corner = cvt([pedestrian.right, pedestrian.bottom], img_scale)
        bottom_left_corner = cvt([pedestrian.left, pedestrian.bottom], img_scale)

        # Creation of the 2 horizontal lines
        cv2.line(
            left_display,
            (int(top_left_corner[0]), int(top_left_corner[1])),
            (int(top_right_corner[0]), int(top_right_corner[1])),
            base_color,
            line_thickness,
        )
        cv2.line(
            left_display,
            (int(bottom_left_corner[0]), int(bottom_left_corner[1])),
            (int(bottom_right_corner[0]), int(bottom_right_corner[1])),
            base_color,
            line_thickness,
        )
        # Creation of 2 vertical lines
        draw_vertical_line(
            left_display,
            bottom_left_corner,
            top_left_corner,
            base_color,
            line_thickness,
        )
        draw_vertical_line(
            left_display,
            bottom_right_corner,
            top_right_corner,
            base_color,
            line_thickness,
        )

        # SHOW BODY KEYPOINTS
        if (show_bodies == True):
            # Draw skeleton bones
            for part in sl.BODY_BONES_POSE_34:
                kp_a = cvt(pedestrian.keypoint_2d[part[0].value], img_scale)
                kp_b = cvt(pedestrian.keypoint_2d[part[1].value], img_scale)
                # Check that the keypoints are inside the image
                if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0]
                        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
                        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0):
                    cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), base_color, 1, cv2.LINE_AA)

            # Scaled ROI
        roi_height = int(top_right_corner[0] - top_left_corner[0])
        roi_width = int(bottom_left_corner[1] - top_left_corner[1])
        overlay_roi = overlay[
            int(top_left_corner[1]): int(top_left_corner[1] + roi_width),
            int(top_left_corner[0]): int(top_left_corner[0] + roi_height),
        ]

        overlay_roi[:, :, :] = base_color

        """
        # Display Object label as text
        position_image = get_image_position(obj.bounding_box_2d, img_scale)
        text_position = (int(position_image[0] - 20), int(position_image[1] - 12))
        text = str(obj.label)
        text_color = (255, 255, 255, 255)
        cv2.putText(
            left_display,
            text,
            text_position,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.5,
            text_color,
            1,
        )

        # Diplay Object distance to camera as text
        if np.isfinite(obj.position[2]):
            text = str(round(abs(obj.position[2]), 1)) + "M"
            text_position = (int(position_image[0] - 20), int(position_image[1]))
            cv2.putText(
                left_display,
                text,
                text_position,
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.5,
                text_color,
                1,
            )
        """
    # Here, overlay is as the left image, but with opaque masks on each detected objects
    cv2.addWeighted(left_display, 0.7, overlay, 0.3, 0.0, left_display)


# ----------------------------------------------------------------------
#       2D TRACKING VIEW
# ----------------------------------------------------------------------


class TrackingViewer:
    def __init__(self, res, fps, D_max):
        # Window size
        self.window_width = res.width
        self.window_height = res.height

        # Visualisation settings
        self.has_background_ready = False
        self.background = np.full(
            (self.window_height, self.window_width, 4), [245, 239, 239, 255], np.uint8
        )

        # Invert Z due to Y axis of ocv window
        # Show objects between [z_min, 0] (z_min < 0)
        self.z_min = -D_max
        # Show objects between [x_min, x_max]
        self.x_min = self.z_min
        self.x_max = -self.x_min

        # Conversion from world position to pixel coordinates
        self.x_step = (self.x_max - self.x_min) / self.window_width
        self.z_step = abs(self.z_min) / (self.window_height)
        # self.z_step = self.x_step

        self.camera_calibration = sl.CalibrationParameters()

        # List of alive tracks
        self.tracklets = []

    def set_camera_calibration(self, calib):
        self.camera_calibration = calib
        self.has_background_ready = False

    def generate_view(
        self,
        pedestrians,
        current_camera_pose,
        tracking_view,
    ):

        # To get position in WORLD reference
        for pedestrian in pedestrians:
            tmp_pos = sl.Translation()
            tmp_pos.init_vector(pedestrian.x, pedestrian.y, pedestrian.z)
            new_pos = (
                tmp_pos * current_camera_pose.get_orientation()
            ).get() + current_camera_pose.get_translation().get()
            pedestrian.position = np.array([new_pos[0], new_pos[1], new_pos[2]])

        # Initialize visualisation
        if not self.has_background_ready:
            self.generate_background()

        np.copyto(tracking_view, self.background, "no")

        # * draw points

        # First add new points and remove the ones that are too old
        # current_timestamp = objects.timestamp.get_seconds()
        # self.add_to_tracklets(objects, current_timestamp)
        # self.prune_old_points(current_timestamp)

        # self.draw_points(objects.object_list,
        #                  tracking_view, current_camera_pose)

        # ! draw orientations
        # self.draw_orientations(tracking_view, current_camera_pose, orientations, pose_trackers)

        # self.draw_orientations_objects(objects.object_list,
        #                                tracking_view,  current_camera_pose, orientations)
        # Draw all tracklets
        # self.draw_tracklets(tracking_view, current_camera_pose)

        self.draw_peds(pedestrians, tracking_view, current_camera_pose)
        # self.draw_grid(pedestrians, tracking_view, current_camera_pose)

    def add_to_tracklets(self, objects, current_timestamp):
        for obj in objects.object_list:
            if (
                (obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK)
                or (not np.isfinite(obj.position[0]))
                or (obj.id < 0)
            ):
                continue

            new_object = True
            for i in range(len(self.tracklets)):
                if self.tracklets[i].id == obj.id:
                    new_object = False
                    self.tracklets[i].add_point(obj, current_timestamp)

            # In case this object does not belong to existing tracks
            if new_object:
                self.tracklets.append(Tracklet(obj, obj.label, current_timestamp))

    def prune_old_points(self, ts):
        track_to_delete = []
        for it in self.tracklets:
            if (ts - it.last_timestamp) > (3):
                track_to_delete.append(it)

        for it in track_to_delete:
            self.tracklets.remove(it)

    # ----------------------------------------------------------------------
    #       Drawing functions
    # ----------------------------------------------------------------------

    def draw_peds(self, pedestrians, tracking_view, current_camera_pose):
        for pedestrian in pedestrians:
            if not np.isfinite(pedestrian.position[0]):
                continue
            clr = generate_color_from_id(pedestrian.track_id)
            pt = TrackPoint(pedestrian.position)
            cv_pos = self.to_cv_point(pt.get_xyz(), current_camera_pose)
            cv_car_pos = self.to_cv_point([0., 0., 0.], current_camera_pose)
            cv_heading = self.to_cv_point([pt.get_xyz()[0] + 1000. * pedestrian.v_x, pt.get_xyz()[1] + 1000. * pedestrian.v_y, pt.get_xyz()[2] + 1000. * pedestrian.v_z], current_camera_pose)
            # convert to tuple
            cv_point = (int(cv_pos[0]), int(cv_pos[1]))
            cv_car = (int(cv_car_pos[0]), int(cv_car_pos[1]))

            cv_heading_point = (int(cv_heading[0]), int(cv_heading[1]))

            cv2.circle(tracking_view, cv_point, 4, clr, 4)

            # cv2.circle(tracking_view, cv_heading_point, 2, clr, 2)

            """
            text = "X: " + str(round(obj.position[0], 1)) + "M"
            text += " Y: " + str(round(obj.position[1], 1)) + "M"
            text += " Z: " + str(round(obj.position[2], 1)) + "M"
            text_color = (0, 0, 0, 255)
            cv2.putText(tracking_view, text, (int(cv_start_point[0]), int(
                cv_start_point[1])),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, text_color, 1)
            """

            # * draw orientations
            # ! correction angle is only needed when the orientation value is calculated by a network (MEBOW), zed model gives the correct values
            # angle_from_camera = int(np.arctan(pt.get_xyz()[0] / pt.get_xyz()[2]) * 180.0 / np.pi)
            angle_from_camera = 0
            arrowlen = 30.0
            ori = pedestrian.orientation
            ori += angle_from_camera
            ori *= np.pi / 180.0

            cv2.arrowedLine(tracking_view, cv_point,
                            (int((cv_pos[0] - np.sin(ori) * arrowlen)), int((cv_pos[1] - np.cos(ori) * arrowlen)),),
                            clr, 5,)

            # * draw pedestrian speed vector
            # factor = 10.
            # cv2.arrowedLine(tracking_view, cv_point,
            #                 (int((cv_pos[0] + factor * pedestrian.v_x)), int((cv_pos[1] + factor * pedestrian.v_z)),), clr, 9)
            # print(pedestrian.v_x)
            # print(pedestrian.v_z)
            # print(pedestrian.v_x / pedestrian.v_z)
            # print(np.arctan(pedestrian.v_x / pedestrian.v_z))
            if (np.isnan(np.arctan(pedestrian.v_x / pedestrian.v_z))):
                speed_vector_angle = int(0)
            else:
                speed_vector_angle = int(np.arctan(pedestrian.v_x / pedestrian.v_z) * 180.0 / np.pi)

            # draw line representing the current heading
            # cv2.line(tracking_view, cv_point, cv_heading_point, clr, 2)

            # * draw vector towards car
            # cv2.arrowedLine(tracking_view, cv_point, cv_car, clr, 5)

            # * draw info
            text_color = (0, 0, 0, 255)
            danger_color = (0, 0, 255, 255)
            warning_color = (0, 215, 255, 255)
            safe_color = (0, 255, 0, 255)

            clr = danger_color
            if (pedestrian.safety_level > 0.2):
                clr = warning_color
            if (pedestrian.safety_level > 0.5):
                clr = safe_color

            # * draw safety color
            # cv2.circle(tracking_view, cv_point, 10, clr, 4)

            # text = str(pedestrian.collision_level)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            text = str(pedestrian.track_id)
            cv2.putText(tracking_view, text, (int(cv_pos[0]) + 0, int(cv_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
            # text = pedestrian.location
            # if (pedestrian.location == "unknown" or pedestrian.location == "road"):
            #     clr = danger_color
            # if (pedestrian.location == "edge"):
            #     clr = warning_color
            # if (pedestrian.location == "safe"):
            #     clr = safe_color
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr)
            # text = str(pedestrian.orientation)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
            # text = str(angle_from_camera)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            # text = str(speed_vector_angle)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            # text = str(pedestrian.awareness_level)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            # text = str(pedestrian.location_level)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

            # text = str(pedestrian.collision_level)
            # cv2.putText(tracking_view, text, (int(cv_pos[0]) + 10, int(cv_pos[1]) + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    def draw_grid(self, pedestrians, tracking_view, current_camera_pose):
        clr = (0, 0, 0, 255)
        for i in range(10):
            for j in range(10):
                pos = self.to_cv_point([i - 5., 0., j - 5.], current_camera_pose)
                cv_pos = (int(pos[0]), int(pos[1]))
                cv2.circle(tracking_view, cv_pos, 2, clr, 2)

    def draw_tracklets(self, tracking_view, current_camera_pose):
        for track in self.tracklets:
            clr = generate_color_id_u(track.id)
            cv_start_point = self.to_cv_point(
                track.positions[0].get_xyz(), current_camera_pose
            )
            for point_index in range(1, len(track.positions)):
                cv_end_point = self.to_cv_point(
                    track.positions[point_index].get_xyz(), current_camera_pose
                )
                cv2.line(
                    tracking_view,
                    (int(cv_start_point[0]), int(cv_start_point[1])),
                    (int(cv_end_point[0]), int(cv_end_point[1])),
                    clr,
                    3,
                )
                cv_start_point = cv_end_point
            cv2.circle(
                tracking_view,
                (int(cv_start_point[0]), int(cv_start_point[1])),
                6,
                clr,
                -1,
            )

    def generate_background(self):
        camera_color = [255, 230, 204, 255]

        # Get FOV intersection with window borders
        fov = 2.0 * math.atan(
            self.camera_calibration.left_cam.image_size.width
            / (2.0 * self.camera_calibration.left_cam.fx)
        )

        z_at_x_max = self.x_max / math.tan(fov / 2.0)
        left_intersection_pt = self.to_cv_point(self.x_min, -z_at_x_max)
        right_intersection_pt = self.to_cv_point(self.x_max, -z_at_x_max)

        # Drawing camera
        camera_pts = np.array(
            [
                left_intersection_pt,
                right_intersection_pt,
                [int(self.window_width / 2), self.window_height],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(self.background, camera_pts, camera_color)

    def to_cv_point(self, x, z):
        out = []
        if isinstance(x, float) and isinstance(z, float):
            out = [
                int((x - self.x_min) / self.x_step),
                int((z - self.z_min) / self.z_step),
            ]
        elif isinstance(x, list) and isinstance(z, sl.Pose):
            # Go to camera current pose
            rotation = z.get_rotation_matrix()
            rotation.inverse()
            tmp = x - (z.get_translation() * rotation.get_orientation()).get()
            new_position = sl.Translation()
            new_position.init_vector(tmp[0], tmp[1], tmp[2])
            out = [
                int(((new_position.get()[0] - self.x_min) / self.x_step) + 0.5),
                int(((new_position.get()[2] - self.z_min) / self.z_step) + 0.5),
            ]
        elif isinstance(x, TrackPoint) and isinstance(z, sl.Pose):
            pos = x.get_xyz()
            out = self.to_cv_point(pos, z)
        else:
            print("Unhandled argument type")
        return out


class TrackPoint:
    def __init__(self, pos_):
        self.x = pos_[0]
        self.y = pos_[1]
        self.z = pos_[2]

    def get_xyz(self):
        return [self.x, self.y, self.z]


class Tracklet:
    def __init__(self, obj_, type_, timestamp_):
        self.id = obj_.id
        self.object_type = type_
        self.positions = deque()
        self.add_point(obj_, timestamp_)

    def add_point(self, obj_, timestamp_):
        self.positions.append(TrackPoint(obj_.position))
        self.last_timestamp = timestamp_
