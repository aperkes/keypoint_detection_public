#!/bin/bash
trap break INT

FILES=/data/birds/postures/birdview-2018/*wav.mp4

base_dir="/data/birds/postures/birdview-2018/"
#base_dir="/data/tmp/"
calib_dir="/data/birds/postures/calibrations/birdview/"

for f in $FILES
    do
        echo "Processing $f..."
        file_name=$(basename "$f")
        #file_trim=${file_name%.w*}
        file_trim=`echo $file_name | cut -d'_' -f1`
        calib_name=$calib_dir$file_trim"/calibration.yaml"
         
        echo $calib_name
        #python /home/ammon/Documents/Scripts/keypoint_detection/eval.py --data_dir=$base_dir --video=$f --checkpoint=/home/ammon/Documents/Scripts/keypoint_detection/models/keypoint_model_checkpoint.pt --calib=$calib_name
        python /home/ammon/Documents/Scripts/keypoint_detection/eval.py --data_dir=$base_dir --video=$file_trim.mp4 --calib=$calib_name
    done
