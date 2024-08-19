#!/bin/bash

CFG='yolov5m.yaml'
EPOCHS=100
BATCH_SIZE=8
IMGSZ="1224"
DEVICE=0
PROJECT='runs/yolo_100'


# train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--imgsz IMGSZ]
                # [--rect] [--resume [RESUME]] [--nosave] [--noval] [--noautoanchor] [--noplots] [--evolve [EVOLVE]] [--bucket BUCKET]
                # [--cache [CACHE]] [--image-weights] [--device DEVICE] [--multi-scale] [--single-cls] [--optimizer {SGD,Adam,AdamW}]
                # [--sync-bn] [--workers WORKERS] [--project PROJECT] [--name NAME] [--exist-ok] [--quad] [--cos-lr]
                # [--label-smoothing LABEL_SMOOTHING] [--patience PATIENCE] [--freeze FREEZE [FREEZE ...]] [--save-period SAVE_PERIOD]
                # [--seed SEED] [--local_rank LOCAL_RANK] [--entity ENTITY] [--upload_dataset [UPLOAD_DATASET]]
                # [--bbox_interval BBOX_INTERVAL] [--artifact_alias ARTIFACT_ALIAS]


train() {
    echo "Name: $NAME, Data:$DATA"

    python3 ../yolov5/train.py \
        --weights $WEIGHTS \
        --cfg $CFG \
        --data  $DATA \
        --epochs $EPOCHS \
        --batch-size=$BATCH_SIZE \
        --img-size=$IMGSZ \
        --rect \
        --device $DEVICE \
        --project $PROJECT \
        --name $NAME \
        --exist-ok

    echo ""
}

test() {
    MAX_DET=10
    TASK='test'    #train, val, test, speed or study

    time python3 ../yolov5/val.py \
    --data $DATA \
    --weights $WEIGHTS \
    --batch-size $BATCH_SIZE \
    --imgsz $IMGSZ \
    --max-det $MAX_DET \
    --task $TASK \
    --device $DEVICE \
    --single-cls \
    --save-txt \
    --save-conf \
    --save-json \
    --project $PROJECT \
    --name $NAME \
    --exist-ok
    # --conf-thres CONF_THRES
    # --iou-thres IOU_THRES

}


NAME='potato_rgb'
DATA='./config/data/potato_rgb.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_rgb_test'
test

NAME='potato_pol'
DATA='./config/data/potato_pol.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_pol_test'
test

NAME='potato_rgbdif'
DATA='./config/data/potato_rgbdif.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_rgbdif_test'
test

NAME='potato_mono'
DATA='./config/data/potato_mono.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_mono_test'
test

NAME='potato_dolp'
DATA='./config/data/potato_dolp.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_dolp_test'
test

NAME='potato_pauli'
DATA='./config/data/potato_pauli.yaml'
WEIGHTS='yolov5m.pt'
train
WEIGHTS="$PROJECT/$NAME/weights/best.pt"
NAME='potato_pauli_test'
test
