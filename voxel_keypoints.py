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
    keypoints = np.zeros((heatmaps.shape[1], 3))
    #import ipdb
    #ipdb.set_trace()
    for kpt in range(heatmaps.shape[1]):
        voxel_grid = np.zeros(grid_size)
        for i,x in enumerate(np.arange(DIM_x[0],DIM_x[1],RES)):
            for j, y in enumerate(np.arange(DIM_y[0],DIM_y[1],RES)):
                for k, z in enumerate(np.arange(DIM_z[0],DIM_z[1],RES)):
                    checks = np.zeros(4)
                    for c in range(4):
                        heatmaps_c = heatmaps[c,kpt]
                        point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                        reproj = np.matmul(P[c],point_3d/point_3d[-1])
                        reproj = reproj[:2] / reproj[-1]
                        u,v = int(reproj[0]),int(reproj[1])
                        if u < 0 or v < 0:
                            break
                        elif u >= DIM_u or v >= DIM_v:
                            break
## NOTE: v,u seems right here, but it could obviously be wrong.
                        voxel_grid[i,j,k] += heatmaps_c[v,u]
                        #ind = np.argmax(voxel_grid)
        # add .5 to place it in the voxel center
## This line is meaty: get the max voxel, unravel it to get xyz, multiply it to scale to real space and add .5 * RES to place in the center of voxel.
        (x,y,z) = np.array(np.unravel_index(np.argmax(voxel_grid),voxel_grid.shape)) * RES + .5 * RES
        #z = RES * (ind // grid_size[2] + 0.5)
        #y = RES * ((ind // grid_size[2]) % grid_size[1] + 0.5)
        #x = RES * (ind // (grid_size[1] * grid_size[2]) + 0.5)
        keypoints[kpt] = [x,y,z]
    return keypoints

## rearranged for iteration
# heatmaps is NUM_KEYPOINTS, 1024, 1024
def voxel_keypoints2(heatmaps,calib_file,count=0,res=RES,grids=[[[250,250,250] * 20],500]):
    old_res = grids[1]
    try:
        print('getting params')
        _,_,P,_ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return
## Figure out resolution and block count:
## Is heatmaps *always* 1024? what about bv2, which isn't square?
    dim_v,dim_u = np.shape(heatmaps[0,0])
    course_dims = np.shape(heatmaps[0,0]) // res 
# This assumes that the heatmaps are square...
    ## Downsample the heatmap according to resolution: 
    course_maps = []
    ## For every point in space
    print('lopping through the stuff')
    grid_size = tuple([d // RES for d in DIMS])
    keypoints = np.zeros((heatmaps.shape[1], 3))
    #import ipdb
    #ipdb.set_trace()
    for kpt in range(heatmaps.shape[1]):
        grid_center = grids[0][kpt]
        grid_size = tuple([d // res for d in grid])
        for m in heatmaps[:,kpt]:
            course_maps.append(block_reduce(m,(course_dims),np.max))
        voxel_grid = np.zeros(grid_size)
        dim_x = [grid_center - old_res / 2,grid_center + old_res / 2]
        dim_y = [grid_center - old_res / 2,grid_center + old_res / 2]
        dim_z = [grid_center - old_res / 2,grid_center + old_res / 2]
        for i,x in enumerate(np.arange(dim_x[0],dim_x[1],res)):
            for j, y in enumerate(np.arange(dim_y[0],dim_y[1],res)):
                for k, z in enumerate(np.arange(dim_z[0],dim_z[1],res)):
                    checks = np.zeros(4)
                    for c in range(4):
                        heatmaps_c = course_maps[c]
                        #heatmaps_c = heatmaps[kpt, c]
                        point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                        reproj = np.matmul(P[c],point_3d/point_3d[-1])
                        reproj = reproj[:2] / reproj[-1]
                        u,v = int(reproj[0]),int(reproj[1])
                        u_c = u // course_dims[0]
                        u_c = v // course_dims[0] 
                        if u < 0 or v < 0:
                            break
                        elif u >= dim_u or v >= dim_v:
                            break
## There's a chance these indices are reversed
                        voxel_grid[i,j,k] += heatmaps_c[v_c,u_c]
                        #ind = np.argmax(voxel_grid)
        # add .5 to place it in the voxel center
## This line is meaty: get the max voxel, unravel it to get xyz, multiply it to scale to real space and add .5 * RES to place in the center of voxel.
        (x,y,z) = np.array(np.unravel_index(np.argmax(voxel_grid),voxel_grid.shape)) * res + .5 * res
        keypoints[kpt] = [x,y,z]
    return keypoints, res


