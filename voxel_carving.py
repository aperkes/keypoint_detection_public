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

## So far these are hard coded, but DIM_u should probably be extracted
RES = 5 #mm, size of voxels
DIMS = [500,500,500] # Cage dimensions in mm
DIM_x = [0,500]
DIM_y = [0,500]
DIM_z = [0,500]
DIM_u = 1024
DIM_v = 1024
MASK_THRESH = .1

def voxel_carving(masks,calib_file,count=0,plot_me=False):
    dim_v,dim_u = np.shape(masks[0])
    try:
        print('getting params')
        _,_,P,_ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return 
   ## For every point in space
    print('lopping through the stuff')
    point_cloud = []
    for x in np.arange(DIM_x[0],DIM_x[1],RES):    
        for y in np.arange(DIM_y[0],DIM_y[1],RES):
            for z in range(DIM_z[0],DIM_z[1],RES):
                checks = np.zeros(4)
                for c in range(4):
                    #pdb.set_trace()
                    mask = masks[c]
                    point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                    reproj = np.matmul(P[c],point_3d/point_3d[-1])
                    reproj = reproj[:2] / reproj[-1]
                    u,v = int(reproj[0]),int(reproj[1])
                    #pdb.set_trace()
## IF it's not in the frame, it's not in the mask
                    if u < 0 or v < 0:
                        break
                    elif u >= dim_u or v >= DIM_v:
                        break 
                    elif mask[v,u] >= MASK_THRESH:
                        checks[c] = 1
                        continue
## If it isn't in the mask, skip to the next voxel
                    else:
                        break
## If you've made it to the end without breaking, add it to the point cloud
                if np.sum(checks) == 4:
                    #print('found a point!')
                    #pdb.set_trace()

                    point_cloud.append(point_3d[:3])

    point_cloud = np.array(point_cloud)
    meta_data = {
        'n_points':len(point_cloud),
        'volume':0,
        'Angle':None,
        'Spread':None
    }
    if plot_me:
        plot_cloud(point_cloud,count)
    return point_cloud, meta_data

## Plots point cloud and saves it as an image (with id 'n')
def plot_cloud(point_cloud,n,meta_data = None):
    if len(point_cloud) > 0:
        print('plotting things')
        color_map = cm.get_cmap('viridis')
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        if False: # Down sampling isn't really necessary, and would mess up my volume calculation. 
            sub_points = random.sample(range(len(point_cloud)),2000)
        else:
            sub_points = range(len(point_cloud))
        z_max,z_min = np.max(point_cloud[:,2]),np.min(point_cloud[:,2])
        z_range = z_max-z_min
        for p in sub_points:
            height_ratio = (point_cloud[p,2] - z_min) / z_range
            #ax.scatter(sub_points[p,0],sub_points[p,1],sub_points[p,2],alpha=.5)
            ax.scatter(point_cloud[p,0],point_cloud[p,1],point_cloud[p,2],alpha=.3,color=color_map(height_ratio))
        file_name = './clouds/frame_' + '%04d'%n + '.png'
        ax.set_xlim([0,400])
        ax.set_ylim([0,400])
        ax.set_zlim([0,400])
        ax.w_xaxis.set_pane_color([0,0,0,1])
        ax.w_yaxis.set_pane_color([0,0,0,1])
        ax.w_zaxis.set_pane_color([0,0,0,1])
        fig.tight_layout()
        fig.savefig(file_name)
        plt.close(fig) 
    return

def round_by(x,m):
    return (x // m) * m

# AS above, but iterates through at a rough resolution first
def voxel_carving_iterative(masks,calib_file,count=0,plot_me=False):
    dim_v,dim_u = np.shape(masks[0])
    try:
        print('getting params')
        _,_,P,_ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return 
   ## For every point in space
    print('lopping through the stuff')
    point_cloud = []
## Define the course and fine resolutions.
    n_blocks = 8
    COURSE = DIMS[0] / n_blocks 
    COURSE_mask = round(len(masks[0]) / n_blocks)
    FINE = 5
    blocks = []
    course_masks = []
## Down sample the masks, this appears to be a good method
    for mask in masks:
        course_masks.append(block_reduce(mask,(COURSE_mask,COURSE_mask),np.max))
    for x in np.arange(DIM_x[0] + COURSE / 2,DIM_x[1],COURSE):    
        for y in np.arange(DIM_y[0] + COURSE / 2,DIM_y[1],COURSE):
            for z in np.arange(DIM_z[0] + COURSE / 2,DIM_z[1],COURSE):
                #print(x,y,z)
                checks = np.zeros(4)
                old_checks = np.zeros(4)
                #pdb.set_trace()
                for c in range(4):
                    #pdb.set_trace()
                    #mask = masks[c]
                    mask = course_masks[c]
                    point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                    reproj = np.matmul(P[c],point_3d/point_3d[-1])
                    reproj = reproj[:2] / reproj[-1]
                    u,v = int(round(reproj[0])),int(round(reproj[1]))
                    u_c = int(u / COURSE_mask)
                    v_c = int(v / COURSE_mask)
                    #pdb.set_trace()
## IF it's not in the frame, it's not in the mask
                    if u < 0 or v < 0:
                        break
                    elif u >= dim_u or v >= dim_v:
                        break 
                    elif masks[c][v,u] >= MASK_THRESH:
                        old_checks[c] = 1
                        #pdb.set_trace()
                        pass
                    if mask[v_c,u_c] >= MASK_THRESH:
                        checks[c] = 1
                        continue
## If it isn't in the mask, skip to the next voxel
                    else:
                        break
## If you've made it to the end without breaking, add it to the point cloud
                if np.sum(checks) == 4:
                    #print('found a point!')
                    #pdb.set_trace()
                    blocks.append((x,y,z))
                    #point_cloud.append(point_3d[:3])

    checked_points = {} 
## Need to go by even digits, even if the blocks came from weird places...a bit hacky, but an easy check.
    for block in blocks:
        x0,x1 = round_by(block[0] - COURSE,FINE),round_by(block[0] + COURSE + 5,FINE)
        y0,y1 = round_by(block[1] - COURSE,FINE),round_by(block[1] + COURSE + 5,FINE)
        z0,z1 = round_by(block[2] - COURSE,FINE),round_by(block[2] + COURSE + 5,FINE)
        for x in np.arange(x0,x1,FINE):
            for y in np.arange(y0,y1,FINE):
                for z in np.arange(z0,z1,FINE):
## Try to efficiently skip overlaps, hopefully memory complexity isn't an issue here...
                    if (x,y,z) in checked_points:
                        continue
                    else:
                        checked_points[(x,y,z)] = 1
                    checks = np.zeros(4)
                    for c in range(4):
                        #pdb.set_trace()
                        #mask = masks[c]
                        mask = masks[c]
                        point_3d = np.array([x,y,z,1000]) ## Need this in homogonous coordinates
                        reproj = np.matmul(P[c],point_3d/point_3d[-1])
                        reproj = reproj[:2] / reproj[-1]
                        u,v = int(reproj[0]),int(reproj[1])
                        #pdb.set_trace()
## IF it's not in the frame, it's not in the mask
                        if u < 0 or v < 0:
                            break
                        elif u >= dim_u or v >= DIM_v:
                            break 
                        elif mask[v,u] >= MASK_THRESH:
                            checks[c] = 1
                            continue
## If it isn't in the mask, skip to the next voxel
                        else:
                            break
## If you've made it to the end without breaking, add it to the point cloud
                    if np.sum(checks) == 4:
                        #print('found a point!')
                        #pdb.set_trace()
                        point_cloud.append(point_3d[:3])
                     
    #pdb.set_trace()
    point_cloud = np.array(point_cloud)
    meta_data = {
        'n_points':len(point_cloud),
        'volume':0,
        'angle':None,
        'spread':None
    }
    if plot_me:
        plot_cloud(point_cloud,count)
    return point_cloud, meta_data


if __name__ == "__main__":
    print('Doing stuff')
    masks = np.load('./masks/mask_125.npy')
    calib_file = './test.yaml'
    print('Doing real stuff...')
    #point_cloud,meta_data1 = voxel_carving(masks,calib_file)
    #print('old points:',len(point_cloud))
    point_cloud,meta_data = voxel_carving_iterative(masks,calib_file)
    print('N-points:',meta_data['n_points'])
