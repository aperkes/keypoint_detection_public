import numpy as np
import os
import argparse

def crawl_dir(directory, pattern):
    file_paths = []
    for filename in os.listdir(directory):
        if pattern in filename:
            file_paths.append(filename.split('.')[0])
    file_paths.sort()
    return file_paths

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to folders containing video sequences')
parser.add_argument('--out_dir', type=str, required=True, help='Path to save csv files')
if __name__ == '__main__':
    args = parser.parse_args()
    videos = crawl_dir(args.data_dir, pattern='.mp4')
    for video in videos:
        try:
            k2d = np.load(os.path.join(args.data_dir, video, 'pred_keypoints_2d.npy'))
            k3d = np.load(os.path.join(args.data_dir, video, 'pred_keypoints_3d.npy'))
            k2d = np.reshape(k2d, (-1,4*20*3))
            k3d = np.reshape(k3d, (-1,20*3))
            k = np.concatenate((k3d, k2d), axis=-1)
            posture_name = video.split('/')[-1]
            if not os.path.exists(os.path.join(args.out_dir, posture_name)):
                os.makedirs(os.path.join(args.out_dir, posture_name))
            np.savetxt(os.path.join(args.out_dir, posture_name, 'detections_all.csv'), k, delimiter=',', fmt= '%10.5f')
            np.savetxt(os.path.join(args.out_dir, posture_name, 'detections_3d.csv'), k3d, delimiter=',', fmt='%10.5f')
        except:
            continue
