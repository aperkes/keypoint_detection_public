#!/bin/bash
trap break INT

FILES=/data/birds/postures/birdview2-2019/*wav.mp4

base_dir="/data/birds/postures/birdview2-2019/"
calib_dir="/data/birds/postures/calibrations/birdview-2/calibrations/"

for f in $FILES
    do
        echo "Processing $f..."
        file_name=$(basename "$f")
        file_trim=${file_name%.w*}
        calib_name=$calib_dir$file_trim"/calibration/calibration.yaml"
        python /home/ammon/Documents/Scripts/keypoint_detection/eval.py --data_dir=$base_dir --video=$f --checkpoint=/home/ammon/Documents/Scripts/keypoint_detection/models/keypoint_model_checkpoint.pt --calib=$calib_path
    done
