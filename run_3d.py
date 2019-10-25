#! /usr/bin/env python

import argparse
import compute_3d_pose

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default=None, required=True, help='Directory containing the 2d_keypoints file')
parser.add_argument('--calib_file',default=None, required=True, help='Calibration file, stored on birdview servers')

args = parser.parse_args()


compute_3d_pose.compute_3d_pose(args.data_dir,calib_file = args.calib_file)
