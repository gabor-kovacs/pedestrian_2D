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

show2Dviewer = True

# Read opts
opts = Opts()
# Prepare
color = np.random.randint(0, 255, (100000, 3))  # Create some random colors
total_frame_num = 0
total_time = 0

cap = cv2.VideoCapture("/home/gabor/Documents/FairMOT/videos/example.mp4")

# Init tracker
tracker = Tracker(opts.max_age, opts.max_dist_iou, opts.max_dist_feature, opts.n_init, opts.reinit_interval, opts.point_termi, opts.start_ind,
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
# model = torch.nn.DataParallel(model)
segmentation_model.to(torch.device('cuda'))
segmentation_model = segmentation_model.eval()
transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

frame_ind = 1

dt = 1.0 / 30.0

while(cap.isOpened()):
    _, frame = cap.read()

    if frame_ind == 1 or frame_ind % opts.show_interval == 0:
        print("Frame: {:06d}".format(frame_ind))

    # Read frame
    # frame = cv2.imread("{}/{:06d}.jpg".format(input_image_dir, frame_ind))

    # tim = time()

    # skip bad frames:
    # if frame is None:
    #     print(f"Dropped frame {frame_ind}")
    #     continue

    tracker.update_frame(frame_ind, frame)
    tracker.update_status_every()
    tracker.update()


    # * Pedestrian orientation
    ped_tensor_list = []
    ped_id_list = []


    for track in tracker.tracks:
        bbox = [int(i) for i in track.bbox]
        if bbox[0] >= 0 and bbox[1] >= 0: # ? DO WE NEED IT?
            w = bbox[2]
            h = bbox[3]
            extrapad = max(w,h) // 10
            top = max(bbox[1]  - extrapad,0)
            bottom = min((bbox[1] + bbox[3]) + extrapad, frame.shape[0] - 1)
            left = max(bbox[0]  - extrapad,0)
            right = min((bbox[0] + bbox[2]) + extrapad, frame.shape[1] - 1)

            ped_im = frame[ top : bottom, left : right ]

            # cv2.imshow("ped_im", ped_im)
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


    # * segmentation
    # resize image
    frame_segmentation = cv2.resize(frame, (960, 540))
    # bgr image for opencv
    frame_segmentation_rgb = cv2.cvtColor(frame_segmentation, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        img_tensor = transform(frame_segmentation_rgb).unsqueeze(0)
        img_tensor = img_tensor.to(torch.device('cuda'))
    pred = segmentation_model(img_tensor).max(1)[1].cpu().numpy()[0]  # HW
    colorized_preds = decode_fn(pred).astype('uint8')
    colorized_preds = cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR)

    # Draw and write
    for track in tracker.tracks:
        # if track.state == TrackState.Confirmed:
        if 1:
            track_id = track.track_id
            bbox = [int(i) for i in track.bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color[track_id].tolist(), thickness=2)
            cv2.putText(frame, str(track_id), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[track_id].tolist(), thickness=2)
            for point in track.points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, color[track_id].tolist(), -1)

            # print(f"track_id: {track_id}, bbox: {bbox}")
            # validate bbox values
            # if bbox[0] >= 0 and bbox[1] >= 0:
            #     cv2.imshow(str(track_id),  frame[bbox[1]: bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
            #     cv2.waitKey(0)
            # cv2.imshow(str(track_id), frame[bbox[0]:bbox[0]+bbox[2], bbox[1]: bbox[1]+bbox[3]])
            # out_track.write("{:d},{:d},{:0.2f},{:0.2f},{:0.2f},{:0.2f},-1,-1,-1,-1\n".format(frame_ind, track_id, bbox[0], bbox[1], bbox[2], bbox[3]))

    # Show
    if show2Dviewer:
        height, width = frame.shape[:2]
        # smaller = cv2.resize(frame, (round(width / 4), round(height / 4)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("tracking", frame)
        # cv2.imshow("colorized_preds", colorized_preds)
        cv2.waitKey(1)

    # print(1 / (time() - tim))

    # Next frame
    frame_ind += 1

# Release
cv2.destroyAllWindows()

# Total frame number
total_frame_num += frame_ind

