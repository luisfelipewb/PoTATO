#!/bin/bash

# usage: detect.py [-h] [--weights WEIGHTS [WEIGHTS ...]] [--source SOURCE] [--data DATA] [--imgsz IMGSZ [IMGSZ ...]] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES] [--max-det MAX_DET] [--device DEVICE] [--view-img] [--save-txt]
#                  [--save-conf] [--save-crop] [--nosave] [--classes CLASSES [CLASSES ...]] [--agnostic-nms] [--augment] [--visualize] [--update] [--project PROJECT] [--name NAME] [--exist-ok] [--line-thickness LINE_THICKNESS]
#                  [--hide-labels] [--hide-conf] [--half] [--dnn] [--vid-stride VID_STRIDE]

# optional arguments:
#   -h, --help            show this help message and exit
#   --weights WEIGHTS [WEIGHTS ...]
#                         model path or triton URL
#   --source SOURCE       file/dir/URL/glob/screen/0(webcam)
#   --data DATA           (optional) dataset.yaml path
#   --imgsz IMGSZ [IMGSZ ...], --img IMGSZ [IMGSZ ...], --img-size IMGSZ [IMGSZ ...]
#                         inference size h,w
#   --conf-thres CONF_THRES
#                         confidence threshold
#   --iou-thres IOU_THRES
#                         NMS IoU threshold
#   --max-det MAX_DET     maximum detections per image
#   --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
#   --view-img            show results
#   --save-txt            save results to *.txt
#   --save-conf           save confidences in --save-txt labels
#   --save-crop           save cropped prediction boxes
#   --nosave              do not save images/videos
#   --classes CLASSES [CLASSES ...]
#                         filter by class: --classes 0, or --classes 0 2 3
#   --agnostic-nms        class-agnostic NMS
#   --augment             augmented inference
#   --visualize           visualize features
#   --update              update all models
#   --project PROJECT     save results to project/name
#   --name NAME           save results to project/name
#   --exist-ok            existing project/name ok, do not increment
#   --line-thickness LINE_THICKNESS
#                         bounding box thickness (pixels)
#   --hide-labels         hide labels
#   --hide-conf           hide confidences
#   --half                use FP16 half-precision inference
#   --dnn                 use OpenCV DNN for ONNX inference
#   --vid-stride VID_STRIDE
#                         video frame-rate stride

IMGSZ="1224"
PROJECT="test_detect"
DEVICE=0
MAX_DET=10
TASK='test'    #train, val, test, speed or study
CONF_THRES=0.1

NAME='potato_pol'
DATA='./config/data/potato_pol.yaml'
WEIGHTS="runs/yolo_100/$NAME/weights/best.pt"
NAME='potato_pol_test'

python3 ../yolov5/detect.py \
    --source "../utils/input/test_images/*_pol.png" \
    --weights $WEIGHTS \
    --imgsz $IMGSZ \
    --max-det $MAX_DET \
    --device $DEVICE \
    --project $PROJECT \
    --name $NAME \
    --conf-thres $CONF_THRES \
    # --save-crop


NAME='potato_rgb'
DATA='./config/data/potato_rgb.yaml'
WEIGHTS="runs/yolo_100/$NAME/weights/best.pt"
NAME='potato_rgb_test'

python3 ../yolov5/detect.py \
    --source "../utils/input/test_images/*_rgb.png" \
    --weights $WEIGHTS \
    --imgsz $IMGSZ \
    --max-det $MAX_DET \
    --device $DEVICE \
    --project $PROJECT \
    --name $NAME \
    --conf-thres $CONF_THRES \
    # --save-crop
