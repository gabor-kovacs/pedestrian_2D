import numpy as np
from numba import jit
import cv2

@jit(nopython=True)
def zed_img_to_bgr(zed_im):
    bgr = zed_im[:, :, 0:3] 
    im = np.copy(bgr)
    return im

@jit(nopython=True)
def zed_point_cloud_to_xyz(pc):
    xyz = pc[:, :, 0:3] 
    return xyz

# @jit(nopython=True)


def get_3d_position_from_xyz(xyz,top,bottom,left,right, mask=None):

    # check if mask contains enough pixels
    ped_pixels = np.count_nonzero(mask)

    if mask is None:
        x_list = xyz[top:bottom,left:right,0].flatten()
        y_list = xyz[top:bottom,left:right,1].flatten()
        z_list = xyz[top:bottom,left:right,2].flatten()
    elif ped_pixels > 64: #at least 8x8 pixels
        x_list = xyz[top:bottom,left:right,0]
        y_list = xyz[top:bottom,left:right,1]
        z_list = xyz[top:bottom,left:right,2]
        x_list = cv2.bitwise_and(x_list, x_list, mask=mask).flatten()
        y_list = cv2.bitwise_and(y_list, y_list, mask=mask).flatten()
        z_list = cv2.bitwise_and(z_list, z_list, mask=mask).flatten()
    else:
        x_list = xyz[top:bottom,left:right,0].flatten()
        y_list = xyz[top:bottom,left:right,1].flatten()
        z_list = xyz[top:bottom,left:right,2].flatten()


    # print(f"size of x_list: {x_list.shape}")
    # print(f"size of y_list: {y_list.shape}")
    # print(f"size of z_list: {z_list.shape}")

    x_filtered  = x_list[np.isfinite(x_list)]
    y_filtered  = y_list[np.isfinite(y_list)]
    z_filtered  = z_list[np.isfinite(z_list)]
    x_filtered  = x_filtered[x_filtered != 0]
    y_filtered  = y_filtered[y_filtered != 0]
    z_filtered  = z_filtered[z_filtered != 0]
    # estimated 3d position of person
    x_median = np.median(x_filtered)
    y_median = np.median(y_filtered)
    z_median = np.median(z_filtered)
    # print(f"{x_median}, {y_median}, {z_median}")
    return x_median, y_median, z_median