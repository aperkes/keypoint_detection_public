#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import block_reduce
from compute_3d_pose import get_camera_params
import pdb
import random

## All I need is calibration and the masks

RES = 10 #mm, size of voxels
DIMS = [500,500,500] # Cage dimensions in mm
DIM_x = [0,500]
DIM_y = [0,500]
DIM_z = [0,500]
DIM_u = 1024
DIM_v = 1024
MASK_THRESH = .1

# heatmaps is NUM_KEYPOINTS, 1024, 1024
def voxel_keypoints(heatmaps,calib_file,count=0):
    try:
        print('getting params')
        _,_,P,_ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return
   ## For every point in space
    print('lopping through the stuff')
    grid_size = tuple([d // RES for d in DIMS])
    keypoints = np.zeros((heatmaps.shape[0], 3))
    for kpt in range(heatmaps.shape[0]):
        voxel_grid = np.zeros(grid_size)
        for x in np.arange(DIM_x[0],DIM_x[1],RES):
            for y in np.arange(DIM_y[0],DIM_y[1],RES):
                for z in range(DIM_z[0],DIM_z[1],RES):
                    checks = np.zeros(4)
                    for c in range(4):
                        heatmaps_c = heatmaps[kpt, c]
                        point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                        reproj = np.matmul(P[c],point_3d/point_3d[-1])
                        reproj = reproj[:2] / reproj[-1]
                        u,v = int(reproj[0]),int(reproj[1])
                        if u < 0 or v < 0:
                            break
                        elif u >= DIM_u or v >= DIM_v:
                            break
                        voxel_grid[x,y,z] += heatmaps[u,v]
                        ind = np.argmax(voxel_grid)
        z = ind // grid_size[2]
        y = (ind // grid_size[2]) % grid_size[1]
        x = ind // (grid_size[1] * grid_size[2])
        keypoints[kpt] = [x,y,z]
    return keypoints


