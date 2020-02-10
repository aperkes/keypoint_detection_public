import numpy as np
import cv2
from os.path import join
import yaml
from tqdm import tqdm

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
        reproj_error[i] = np.median(np.array(re))
    ind = np.argmin(reproj_error)
#    print('reproj error: ', reproj_error[ind], ' X: ', X[ind][:-1,0])
    return X[ind][:-1,0]

def compute_3d_pose(data_dir, calib_file='/NAS/home/bird_postures/postures_2017/extrinsic_calib.yaml'):
    try:
        _, _, P, _ = get_camera_params(calib_file)
    except:
        print('Calibration file does not exist')
        return
        
    keypoints = np.load(join(data_dir, 'pred_keypoints_2d.npy'))
    X = np.zeros((keypoints.shape[3], keypoints.shape[1], 3))
    for frame in tqdm(range(keypoints.shape[3])):
        for kpt in range(keypoints.shape[1]):
            X[frame, kpt, :] = triangulate_points(keypoints[:,kpt,:,frame], P)
    np.save(join(data_dir, 'pred_keypoints_3d.npy'), X)

