import os

import utils

if __name__ == '__main__':

    args = utils.parse_arguments()

    ann_path = os.path.join(args.folder, 'coco_splits/result.json')
    img_folder = os.path.join(args.folder, 'images')
    tags = ["_rgb", "_pol", "_dolp", "_mono", "_rgbdif"]

    print(f"Annotation  :\t{ann_path}")
    print(f"Image folder:\t{img_folder}")
    print(f"Tags        :\t{tags}")

    # Get all file names from the result.json
    filenames = utils.get_filenames(ann_path, img_folder, tags)

    # Generate sequential split
    output_dir = "./output/split_seq"
    trn_files, val_files, tst_files = utils.split_sequential(filenames, trn_n=2000, val_n=600, tst_n=2000)
    utils.save_txt_split(trn_files, val_files, tst_files, output_dir)
    utils.generate_coco_split(ann_path, trn_files, val_files, tst_files, tags, output_dir)
    utils.generate_yolo_split(trn_files, val_files, tst_files, tags, output_dir)

    # Generate random split
    # output_dir = "./output/split_rand1"
    # trn_files, val_files, tst_files = utils.split_random(filenames, trn_n=2000, val_n=600, tst_n=2000)
    # utils.save_txt_split(trn_files, val_files, tst_files, output_dir)
    # utils.generate_coco_split(ann_path, trn_files, val_files, tst_files, tags, output_dir)
    # utils.generate_yolo_split(trn_files, val_files, tst_files, tags, output_dir)

    print("Done")
