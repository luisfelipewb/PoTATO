#!/bin/bash

LOG='./runs/training.log'
mkdir -p ./runs/

CONFIG_FILES=(
"./config/fasterrcnn.yaml"
"./config/maskrcnn.yaml"
"./config/retinanet.yaml"
)
TAGS=(
"rgb"
"pol"
# "mono"
# "rgb90"
# "rgbdif"
# "dolp"
# "flow"
)

DEVICE='cuda:1'


ANN_FOLDER="../datasets/potato/split_seq"
IMG_FOLDER="../datasets/potato/images/"


for config_file in "${CONFIG_FILES[@]}"
do
    # Run regular detectron detectrion
    for tag in "${TAGS[@]}"
    do
        echo $(date -Is) "Starting $config_file for $tag images " >> $LOG
        start=`date +%s`

        echo "Training: $config_file for $tag images"
        python3 train_potato.py --config $config_file --tag $tag --device $DEVICE --ann_folder $ANN_FOLDER --img_folder $IMG_FOLDER
        end=`date +%s`
        runtime=$((end-start))
        echo $(date -Is) "Finished $config_file for $tag images in $runtime seconds" >> $LOG

        sleep 30

        echo "Evaluation: $config_file for $tag images"
        python3 train_potato.py --config $config_file --tag $tag --device $DEVICE --ann_folder $ANN_FOLDER --img_folder $IMG_FOLDER --eval_only

        sleep 60
    done
done

