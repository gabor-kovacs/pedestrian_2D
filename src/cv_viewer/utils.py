import cv2
import numpy as np
import pyzed.sl as sl


import random
import colorsys

ID_COLORS = [(232, 176, 59),
             (175, 208, 25),
             (102, 205, 105),
             (185, 0, 255),
             (99, 107, 252)]

# Slightly differs from sl.BODY_BONES in order to draw the spine
SKELETON_BONES = [(sl.BODY_PARTS.NOSE, sl.BODY_PARTS.NECK),
                  (sl.BODY_PARTS.NECK, sl.BODY_PARTS.RIGHT_SHOULDER),
                  (sl.BODY_PARTS.RIGHT_SHOULDER, sl.BODY_PARTS.RIGHT_ELBOW),
                  (sl.BODY_PARTS.RIGHT_ELBOW, sl.BODY_PARTS.RIGHT_WRIST),
                  (sl.BODY_PARTS.NECK, sl.BODY_PARTS.LEFT_SHOULDER),
                  (sl.BODY_PARTS.LEFT_SHOULDER, sl.BODY_PARTS.LEFT_ELBOW),
                  (sl.BODY_PARTS.LEFT_ELBOW, sl.BODY_PARTS.LEFT_WRIST),
                  (sl.BODY_PARTS.RIGHT_HIP, sl.BODY_PARTS.RIGHT_KNEE),
                  (sl.BODY_PARTS.RIGHT_KNEE, sl.BODY_PARTS.RIGHT_ANKLE),
                  (sl.BODY_PARTS.LEFT_HIP, sl.BODY_PARTS.LEFT_KNEE),
                  (sl.BODY_PARTS.LEFT_KNEE, sl.BODY_PARTS.LEFT_ANKLE),
                  (sl.BODY_PARTS.RIGHT_SHOULDER, sl.BODY_PARTS.LEFT_SHOULDER),
                  (sl.BODY_PARTS.RIGHT_HIP, sl.BODY_PARTS.LEFT_HIP),
                  (sl.BODY_PARTS.NOSE, sl.BODY_PARTS.RIGHT_EYE),
                  (sl.BODY_PARTS.RIGHT_EYE, sl.BODY_PARTS.RIGHT_EAR),
                  (sl.BODY_PARTS.NOSE, sl.BODY_PARTS.LEFT_EYE),
                  (sl.BODY_PARTS.LEFT_EYE, sl.BODY_PARTS.LEFT_EAR)]


def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))


def generate_color_id_u(idx):
    arr = []
    if(idx < 0):
        arr = [236, 184, 36, 255]
    else:
        color_idx = idx % 5
        arr = [ID_COLORS[color_idx][0], ID_COLORS[color_idx][1], ID_COLORS[color_idx][2], 255]
    return arr


def draw_vertical_line(left_display, start_pt, end_pt, clr, thickness):
    n_steps = 7
    pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps, ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
    pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps, (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]

    cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
    cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)


def generate_color_from_id(id):
    random.seed(id)
    h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
    return [r, g, b, 255]
