import os

import utils

if __name__ == '__main__':

    args = utils.parse_arguments()

    ann_path = os.path.join(args.folder, 'coco_splits/result.json')
    print(f"Annotation:\t{ann_path}")

    input_files=["yolo_dolp.json", "yolo_mono.json", "yolo_pol.json", "yolo_rgbdif.json", "yolo_rgb.json"]

    for det_file in input_files:
        det_path = os.path.join('input/detections/',det_file)
        utils.fix_yolo_detection(ann_path, det_path)
        print()

    print("Done")
