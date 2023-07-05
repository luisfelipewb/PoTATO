# PoTATO - A Dataset for Analyzing Polarimetric Traces of Afloat Trash Objects

## Description 

The PoTATO dataset contains more than 12k labeled plastic bottles with RAW polarimetric and color images.

![TODO: Add GIF](image.jpg)

![TODO: Add image](image.jpg)

This repository contains the scripts and code related to the paper "[TODO: Add link](https://www.example.com)". The code provided here enables the reproduction of the experiments described in the paper.

## Steps for reproducing the experiments

1. Clone this repository:
git clone https://github.com/luisfelipewb/PoTATO.git


2. Download the PoTATO dataset:
```bash
cd ./datasets
wget link-to-dataset
```

3. Extract the images from the RAW data (This might take a while and will need 25GB space). The imagess will be generated in the same folder as the raw files:
```bash
python3 utils/extract_from_raw.py datasets/potato/images_raw/
```

4. Place images in the correct folder:
```bash
images_raw -> images as expected by YOLOv5
```

5. Install YOLO-v5:
Same instructions from the YOLOv5 repository [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
The 'yolov5' folder should be on the same level as the 'experiments' folder
```bash
git clone https://github.com/ultralytics/yolov5
pip install -r ./yolov5/requirements.txt
```

6. Run YOLO training:
```bash
./train_yolo.sh
```

7. Run YOLO detections:
```bash
./detect_yolo.sh
```

8. Install Detectron2: 
Same instructions from the Detectron2 repository [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
The 'detectron2' folder should be on the same level as the 'experiments' folder
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

9. Run Detectron2 training:
```bash
./train_det2.sh
```

10. Fix the `image_id` in yolo detection. The `image_id` used for the YOLO detections does no correspond to the annotation. The following script updates the `image_id` for extracting the COCO metrics
```bash
cd utils
python3 fix_yolo_detections.py
```

11. Generate metrics
```bash
python3 coco_metrics.py
```
The results will be avaiable in `utils/output/metrics.csv`

Please cite the paper when using the dataset or code


