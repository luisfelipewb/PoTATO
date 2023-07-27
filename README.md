# PoTATO - A Dataset for Analyzing Polarimetric Traces of Afloat Trash Objects

## Description 

The PoTATO dataset contains more than 12k labeled plastic bottles with RAW polarimetric and color images.

### Example bottles in the dataset

Example of bottle images and the different channels that can be extracted. Each column has different channels of the same image, where the labels correspond to MONO, RGB, RGB90, RGBDIF, DOLP, and POL.

![](img/samples.jpg)

### Extracted Channels

Example of channels that can be extracted from the RAW images.

![](img/exp05_frame042538_tile.jpg)

### Detection using YOLOv5

The sequence of images illustrates the object detection that can be implemented independently in the RGB or the POL channels.

![](img/detection_sequence.gif)

### Filtering reflection

The reflection of light on the water surface can be filtered due to its polarization properties. Example of RGB, RGB90, and RGBDIF channels.

|![](./img/exp02_frame00609.gif) | ![](./img/exp04_frame01602.gif) |
|:------------------------------:|:-------------------------------:|

### RGB and POL

Comparison between RGB and POL channels, showcasing the pronounced difference in color contrast.

|![](./img/exp06_frame025619.gif) | ![](./img/exp07_frame022868.gif) |
|:-------------------------------:|:----------------------------:|

This repository contains the scripts and code for the paper [TODO](https://www.example.com), enabling the replication of the described experiments and fostering research reproducibility

## Steps for reproducing the experiments

1. Clone this repository:
```bash
git clone https://github.com/luisfelipewb/PoTATO.git
```


2. Download the PoTATO dataset from [dataset link](https://gtvault-my.sharepoint.com/:f:/g/personal/lbatista3_gatech_edu/EsRU8LnjkXZLl7bPbMqJfaIB9YAZCJw5lEy6VplUfn8WnQ?e=1njdbX):
```bash
mv ~/Downloads/potato.tgz ./datasets
cd ./datasets && tar -xzvf potato.tgz
```

3. Extract the images from the RAW data (This might take a while and will need 25GB space). The imagess will be generated in the same folder as the raw files:
```bash
python3 utils/extract_from_raw.py datasets/potato/images_raw/
```

4. Place images in the correct folder:
```bash
mv images_raw images
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


