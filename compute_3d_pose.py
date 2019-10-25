import numpy as np
import cv2
from os.path import join
import yaml
from tqdm import tqdm

def get_camera_params(filename, ncams=4):
    with open(filename) as stream:
        try:
            data = yaml.load(stream)
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
    for i in range(len(tuples)):
        t = tuples[i]
        P1 = P[t[0]]
        P2 = P[t[1]]
        p1 = x[t[0]] 
        p2 = x[t[1]] 
        p3d = cv2.triangulatePoints(P1, P2, p1, p2)
        X.append(p3d/p3d[-1])
        re = []
        for j in range(4):
            reproj = np.matmul(P[j],p3d)
            reproj = reproj[:2] / reproj[-1]
            error = np.sum((reproj - x[j][:, None]) ** 2)
            re.append(error)
        reproj_error[i] = np.median(np.array(re))
    ind = np.argmin(reproj_error)
    return X[ind][:-1,0]

def compute_3d_pose(data_dir, calib_file='/NAS/home/bird_postures/postures_2017/extrinsic_calib.yaml'):
    try:
        _, _, P, _ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return
        
    keypoints = np.load(join(data_dir, 'pred_keypoints_2d.npy'))
    keypoints = np.reshape(keypoints, (-1,4,20,3))
    
    X = np.zeros((keypoints.shape[0], keypoints.shape[2], 3))
    for frame in tqdm(range(keypoints.shape[0])):
        for kpt in range(keypoints.shape[2]):
            X[frame, kpt, :] = triangulate_points(keypoints[frame,:,kpt,:-1], P)
    np.save(join(data_dir, 'pred_keypoints_3d.npy'), X)

