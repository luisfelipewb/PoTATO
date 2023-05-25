import os

import utils

if __name__ == '__main__':

    args = utils.parse_arguments()

    ann_path = os.path.join(args.folder, 'coco_splits/result.json')
    print(f"Annotation:\t{ann_path}")

    # input_files=["pol_fine_tuned.json", "pol_scratch.json", "rgb_fine_tuned.json", "rgb_scratch.json"]
    # input_files=["pol_fine_tuned.json", "pol_scratch.json", "rgb_fine_tuned.json", "rgb_scratch.json"]
    input_files=["early_fusion_fine_tuned.json", "early_fusion_scratch.json", "pol_fine_tuned.json", "pol_scratch.json", "rgb_fine_tuned.json", "rgb_scratch.json"]
    
    for det_file in input_files:
        det_path = os.path.join('input/detections/',det_file)
        utils.fix_yolo_detection(ann_path, det_path)
        print()

    print("Done")