import cv2

class Opts:
    # Tracker
    max_dist_iou = 0.7  # 0.7
    max_dist_feature = 0.3  # 0.3
    max_age = 10  # 5, 10 ##### C
    n_init = 10
    reinit_interval = 1  # 1, 5, 10, 15, 20, 25
    detect_method = "fairmot"  # "read_det" or "read_gt" or "maskrcnn" or "centermask" or "fairmot"
    point_termi = "variance"  # "variance" or "homography"
    show_interval = 100
    start_ind = 1  # start from 1
    end_ind = 100000000
    show_output_images = True
    save_output_images = False
    max_size = 1000
    thre_conf = 0.2  # FairMOT have to set in command line!! 0.2 ## read_det have to set to -2!!
    thre_var_ratio = 10000000  # 10, 10000000 ##### T
    thre_homo = 100000

    save_old_tracks = True

    # Point detection
    point_detect = "auto"  # "auto" or "manual"
    head_detect = 0  # 1 or 0

    # Manual
    r_ratio = 0.3
    interval_num = 2
    K = 8
    focus_point_manual = "head"  # "head" or "center"

    # Auto
    # Mask
    focus_point_auto = "center"  # "head" or "center" or "none"
    use_mask = 0  # 1 or 0 ##### S
    max_point_num = 10  

    # Shi-Tomasi
    shi_tomasi = 0  # 1: shi_tomasi (slow) or 0: random
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.001,
                          minDistance=1,
                          blockSize=7)

    # Lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

