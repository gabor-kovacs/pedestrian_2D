import numpy as np
from numba import jit

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


def get_3d_position_from_xyz(xyz,top,bottom,left,right):
    x_list = xyz[top:bottom,left:right,0].flatten()
    y_list = xyz[top:bottom,left:right,1].flatten()
    z_list = xyz[top:bottom,left:right,2].flatten()
    x_filtered  = x_list[np.isfinite(x_list)]
    y_filtered  = y_list[np.isfinite(y_list)]
    z_filtered  = z_list[np.isfinite(z_list)]
    # estimated 3d position of person
    x_median = np.median(x_filtered)
    y_median = np.median(y_filtered)
    z_median = np.median(z_filtered)
    # print(f"{x_median}, {y_median}, {z_median}")
    return x_median, y_median, z_median