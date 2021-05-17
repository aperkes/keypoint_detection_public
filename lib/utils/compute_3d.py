import numpy as np
import cv2
from os.path import join
import yaml
from tqdm import tqdm
import itertools
from skimage.measure import block_reduce

MASK_THRESH = .1
def get_camera_params(filename, ncams=4):
    with open(filename) as stream:
        try:
            #data = yaml.load(stream)
            data = yaml.load(stream,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # list of camera calibration matrices
    P = []
    K = []
    H = []
    dist_coeffs = []
    for i in range(ncams): 
        # Pi = np.array(data['cam%d' % i]['T_cam_imu'])
        Pi = np.array(data['cam%d' % i]['T_cam_body'])
        intrinsics = data['cam%d' % i]['intrinsics']
        dist_coeffs.append(data['cam%d' % i]['distortion_coeffs'])
        Ki = np.array([[intrinsics[0], 0, intrinsics[2], 0],
                      [0, intrinsics[1], intrinsics[3], 0],
                      [0, 0, 1, 0]])
        P.append(Pi)
        K.append(Ki)
        H.append(np.matmul(Ki,Pi))
    P = np.stack(P, axis=0)
    K = np.stack(K, axis=0)
    H = np.stack(H, axis=0)
    dist_coeffs = np.array(dist_coeffs)
    return P, K, H, dist_coeffs

def triangulate_points(x, P):
    tuples = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    X = []
    reproj_error = np.zeros(6)
    conf = x[:,2]
    for i in range(len(tuples)):
        t = tuples[i]
        P1 = P[t[0]]
        P2 = P[t[1]]
        p1, c1 = x[t[0], 0:2], x[t[0], 2]
        p2, c2 = x[t[1], 0:2], x[t[0], 2]
        p3d = cv2.triangulatePoints(P1, P2, p1, p2)
        X.append(p3d/p3d[-1])
#        print(t[0], t[1], ' uv: ', p1, p2, ' -> ', np.transpose((p3d/p3d[-1])[:-1]))
        re = []
        for j in range(4):
            reproj = np.matmul(P[j],p3d)
            reproj = reproj[:2] / reproj[-1]
            error_raw = np.sum((reproj - x[j, 0:2][:, None]) ** 2)
            error = error_raw * np.exp(2.0*(3.0 - (c1 + c2 + conf[j])))
#            print(error_raw, ' cooked: ', error, ' c1: ', c1, ' c2: ', c2, ' c3: ', conf[j])
            re.append(error)
        reproj_error[i] = np.nanmedian(np.array(re))
    ind = np.argmin(reproj_error)
#    print('reproj error: ', reproj_error[ind], ' X: ', X[ind][:-1,0])
    return X[ind][:-1,0]

def compute_3d_pose(keypoints, calib_file='/NAS/home/bird_postures/postures_2017/extrinsic_calib.yaml',save_me=True):
    try:
        _, _, P, _ = get_camera_params(calib_file)
    except:
        print('Something wrong with calibration file:',calib_file)
        return
        
    if isinstance(keypoints, str):
        data_dir = keypoints
        keypoints = np.load(join(data_dir, 'pred_keypoints_2d.npy'))
    else:
        data_dir = '.'
    X = np.zeros((keypoints.shape[3], keypoints.shape[1], 3))
    for frame in tqdm(range(keypoints.shape[3])):
        for kpt in range(keypoints.shape[1]):
            X[frame, kpt, :] = triangulate_points(keypoints[:,kpt,:,frame], P)
    if save_me:
        np.save(join(data_dir, 'pred_keypoints_3d.npy'), X)
    return X

def round_by(x,m):
    return (x // m) * m

## Outputs pointcloud, can also store pca and/or plot of point cloud
def voxel_carving(masks,calib_file,count=0,res=50,grids = [[[250,250,250]],250],plot_me=False,pca=False,out_dir = '.'):
    DIMS = [500,500,500] ## Rough Dimensions of the cage (mm)
    if len(grids[0]) == 0:
        return ([],res)
    old_res = grids[1]
    try:
        _,_,P,_ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return

    dim_v,dim_u = np.shape(masks[0])
    n_blocks = round(DIMS[0] // res)
    #print('mask shape:',np.shape(masks[0]))
    #cv2.imwrite('test_mask.png',masks[0] * 255)
    course_dims = np.array(np.shape(masks[0])) / (n_blocks)
    #print('n_blocks:',n_blocks)
    #course_dims = [n_blocks,n_blocks]
    
    all_points = []
    for grid in grids[0]:
        x0,x1 = round_by(grid[0] - old_res,res) + res/2,round_by(grid[0] + old_res + res,res)
        y0,y1 = round_by(grid[1] - old_res,res) + res/2,round_by(grid[1] + old_res + res,res)
        z0,z1 = round_by(grid[2] - old_res,res) + res/2,round_by(grid[2] + old_res + res,res)
        xs = np.arange(x0,x1,res)
        ys = np.arange(y0,y1,res)
        zs = np.arange(z0,z1,res)
        all_points.extend(list(itertools.product(xs,ys,zs)))
    all_points = np.array(all_points)
    #all_points = np.unique(all_points,axis=0) 
    hom_points = np.ones([len(all_points),4])
    hom_points[:,:3] = all_points / 1000
    
    reproj_points = []
    course_masks = []
    for c in range(4):
        course_masks.append(block_reduce(masks[c],tuple(course_dims.astype(int)),np.max))
        reproj_points_c = np.dot(P[c],np.transpose(hom_points))
        reproj_points_c = reproj_points_c / reproj_points_c[2,:]
        reproj_points.append(reproj_points_c.astype(int))
    voxel_list = []
    checked_points = {}
    #pdb.set_trace()
    for p in range(len(all_points)):
        checks = 0
        x,y,z = all_points[p]
        if (x,y,z) in checked_points:
            continue
        else:
            checked_points[(x,y,z)] = 1
        for c in range(4):
            mask = course_masks[c]
            #print(mask.shape)
            u,v = reproj_points[c][:2,p]
            if u < 0 or v < 0:
                break
            elif u >= dim_u or v >= dim_v:
                break
            #pdb.set_trace()
            #print(course_dims,u,v)
            u_c = int(u / course_dims[1])
            v_c = int(v / course_dims[0])
            if mask[v_c,u_c] >= MASK_THRESH:
                checks += 1
                if checks == 4:
                    #pdb.set_trace()
                    voxel_list.append(all_points[p])
                continue
            else:
                break
    voxel_list = np.array(voxel_list)
    if pca:
        if len(voxel_list) > 0:

            pca_cloud = PCA()
            pca_cloud.fit(voxel_list)
            vector1,vector2 = pca_cloud.components_[0:2]
            pca_file = out_dir + '/pca.txt'
            out_string = str(count) + ',' + str(vector1.flatten()).replace('[','').replace(' ',',').replace(']','')
        else:
            out_string = str(count) + ',' + '0,0,0'
        with open(pca_file,'a') as f:
            f.write(out_string + '\n')
## Append it to the file
    if plot_me:
        plot_cloud(voxel_list,count,out_dir)
    return (voxel_list,res)

def iterate_voxels(masks,calib_file,count):
    if calib_file == None:
        return 0
    round1 = voxel_carving(masks,calib_file,count=count,res=100)
    round2 = voxel_carving(masks,calib_file,count=count,res=20,grids=round1)
    round3 = voxel_carving(masks,calib_file,count=count,res=5,grids=round2)
    return round3


