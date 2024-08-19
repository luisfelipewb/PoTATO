import os

import utils

if __name__ == '__main__':

    args = utils.parse_arguments()

    ann_path = os.path.join(args.folder, 'split_seq/result.json')
    print(f"Annotation:\t{ann_path}")

    #input_files=["yolo_dolp.json", "yolo_mono.json", "yolo_pol.json", "yolo_rgbdif.json", "yolo_rgb.json"]
    input_files=["potato_dolp_test", "potato_mono_test", "potato_pol_test", "potato_rgbdif_test", "potato_rgb_test"]

    for det_file in input_files:
        det_dir = os.path.join('../experiments/runs/yolo_100/',det_file)
        det_path = os.path.join(det_dir, "best_predictions.json")
        utils.fix_yolo_detection(ann_path, det_path)
        print()

    print("Done")
