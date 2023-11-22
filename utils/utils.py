import os
import glob
import argparse
import random
import json
import numpy as np
import cv2

def get_file_id(file):
    """
    Returns the file ID for a given file.

    The file ID is defined as the file name without the file extension and without the last _xxx tag.
    For example, this transforms ./a/b/exp00_frame00000_rgb.png into exp00_frame00000.
    This id is unique for each image and can be used to get other channels of the same image.

    :param file: path to the file
    :return: file ID
    """
    file_name = os.path.splitext(os.path.basename(file))[0]
    name_parts = file_name.split('_')
    file_id = '_'.join([name_parts[0], name_parts[1]])
    return file_id

def get_filenames(annotation_file, image_folder, tags):
    """
    Retrieve the list of file names associated with the given annotation file.

    Parameters:
        annotation_file (str): Path to the annotation JSON file in the COCO format.
        image_folder (str): Path to the folder containing the images.
        tags (list): List of desired tags to check for corresponding images.

    Returns:
        list: A list of file names (file IDs without the tags) extracted from the annotation file.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ValueError: If any of the desired tags do not have corresponding images.
    """

    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    filenames = []

    for image in annotations['images']:
        file_name = image['file_name']
        file_id = get_file_id(file_name)
        # For changing channels later
        for tag in tags:
            image_path = os.path.join(image_folder, file_id+tag+'.png')
            if os.path.exists(image_path) is False:
                print(f"ERROR: Could not find image {image_path} that is referenced on {annotation_file}")
        filenames.append(file_id)

    return filenames

def split_random(filenames, trn_n=2000, val_n=600, tst_n=2000):
    """
    Randomly splits a list of filenames into training, validation, and testing sets.

    Args:
        filenames (list): A list of filenames to be split.
        trn_n (int, optional): The desired number of files for the training set. Defaults to 2000.
        val_n (int, optional): The desired number of files for the validation set. Defaults to 600.
        tst_n (int, optional): The desired number of files for the testing set. Defaults to 2000.

    Returns:
        tuple: A tuple containing the training, validation, and testing sets as lists of filenames.

    Raises:
        AssertionError: If the actual length of any split does not match the desired length.

    """
    
    all_files = filenames.copy()

    random.shuffle(all_files)
    length = len(all_files)

    idx_trn = trn_n
    idx_val = trn_n + val_n
    idx_tst = idx_val + tst_n

    trn_files = all_files[:idx_trn]
    val_files = all_files[idx_trn:idx_val]
    tst_files = all_files[idx_val:idx_tst]

    len_trn = len(trn_files)
    len_val = len(val_files)
    len_tst = len(tst_files)
    assert trn_n == len_trn , f"trn_n: desired length: {trn_n} does not match acutal length {len_trn}"
    assert val_n == len_val , f"val_n: desired length: {val_n} does not match acutal length {len_val}"
    assert tst_n == len_tst , f"tst_n: desired length: {tst_n} does not match acutal length {len_tst}"

    trn_p = 100*len_trn/length
    val_p = 100*len_val/length
    tst_p = 100*len_tst/length
    print(f"Random split:")
    print(f"  train: {len_trn} ({trn_p:.2f}%)")
    print(f"  val:   {len_val} ({val_p:.2f}%)")
    print(f"  test:  {len_tst} ({tst_p:.2f}%)")

    return trn_files, val_files, tst_files

def group_names_by_prefix(strings):
    """
    Reorganize a sorted 1D array of strings into a 2D array based on their first 5 characters.

    Parameters:
        strings (list): A sorted 1D array of strings.

    Returns:
        list: A 2D array where the strings are grouped based on their first 5 characters.
    """
    groups = []
    current_prefix = None
    current_group = []

    for string in strings:
        prefix = string[:5]
        if prefix != current_prefix:
            if current_group:
                groups.append(current_group)
            current_prefix = prefix
            current_group = [string]
        else:
            current_group.append(string)

    if current_group:
        groups.append(current_group)

    return groups

def split_sequential(filenames, trn_n, val_n, tst_n):
    """
    Splits a list of filenames into training, validation, and testing sets based on given proportions.
    The same proportion is kept on each experiment and the split is sequential to avoid similar images being
    stored on the training and test sets. REQUIREMENT: the filenames must start with expXX for correct grouping.

    Args:
        filenames (list): A list of filenames.
        trn_n (int): The desired number of training samples.
        val_n (int): The desired number of validation samples.
        tst_n (int): The desired number of testing samples.

    Returns:
        tuple: A tuple containing the training filenames, validation filenames, and testing filenames.

    Raises:
        ValueError: If the total number of annotations is less than the sum of desired splits.
        ValueError: If the actual length of any split is different from the desired size.

    """
    # Check if the size of the splits
    ann_size = len(filenames)
    sum_size = trn_n + val_n + tst_n
    if (ann_size < sum_size):
        raise ValueError(f"Total of {ann_size} annotations available, trying to use {sum_size}")

    if (ann_size > sum_size):
        print(f"WARNING: using {sum_size} out of {ann_size} annotations")

    # Compute percentage values
    trn_p = trn_n / ann_size
    val_p = val_n / ann_size
    tst_p = tst_n / ann_size
    print(f"Split proportions: trn {trn_p:.2f} val {val_p:.2f} tst {tst_p:.2f}")

    # Reorganize the 1d array of filenames into a 2d array grouping each experiment by experiment name
    filenames.sort()
    grouped = group_names_by_prefix(filenames)

    trn_files = []
    val_files = []
    tst_files = []

    # The first groups, split approximately based on the proportion.
    for group in grouped[:-1]:
        group_len = len(group)

        idx_trn = int(round(trn_p * group_len))
        idx_val = int(round((val_p * group_len) + idx_trn))

        # Print lenght of each split
        trn_l = len(group[:idx_trn])
        val_l = len(group[idx_trn:idx_val])
        tst_l = len(group[idx_val:])
        print(f"Images from {group[0][:5]} : {group_len} --> {trn_l} {val_l} {tst_l}")

        trn_files += group[:idx_trn]
        val_files += group[idx_trn:idx_val]
        tst_files += group[idx_val:]


    # For the last, make sure the proportions match the desired numbers using the remaining images for each split.
    group = grouped[-1]
    idx_trn = trn_n - len(trn_files)
    idx_val = val_n - len(val_files) + idx_trn

    trn_files += group[:idx_trn]
    val_files += group[idx_trn:idx_val]
    tst_files += group[idx_val:]

    # Print lenght of each split
    trn_l = len(group[0:idx_trn])
    val_l = len(group[idx_trn:idx_val])
    tst_l = len(group[idx_val:])
    print(f"Images from {group[0][:5]} : {len(group)} --> {trn_l} {val_l} {tst_l}")

    # Confirm if final size is matching
    if trn_n != len(trn_files):
        raise ValueError(f"Actual length of {trn_n} is different from desired size: {len(trn_files)}")
    if val_n != len(val_files):
        raise ValueError(f"Actual length of {val_n} is different from desired size: {len(val_files)}")
    if tst_n != len(tst_files):
        raise ValueError(f"Actual length of {tst_n} is different from desired size: {len(tst_files)}")

    print(f"Generated a split with sizes: {len(trn_files)} {len(val_files)} {len(tst_files)}")

    return trn_files, val_files, tst_files




def save_txt_split(trn_files, val_files, tst_files, output_dir="./output"):
    """
    Saves the filenames of the training, validation, and testing sets into separate text files.

    Args:
        trn_files (list): A list of training filenames.
        val_files (list): A list of validation filenames.
        tst_files (list): A list of testing filenames.
        output_dir (str, optional): The output directory path. Defaults to "./output".

    Returns:
        None

    """

    os.makedirs(output_dir, exist_ok=True)

    trn_path = os.path.join(output_dir,"train.txt")
    val_path = os.path.join(output_dir,"val.txt")
    tst_path = os.path.join(output_dir,"test.txt")

    with open(trn_path, "w") as f:
        f.write("\n".join(trn_files))
        f.write("\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val_files))
        f.write("\n")
    with open(tst_path, "w") as f:
        f.write("\n".join(tst_files))
        f.write("\n")

    print(f"Saving files on {output_dir}")

def load_txt_split(trn_file, val_file, tst_file, path):

    trn_path = os.path.join(path, trn_file)
    val_path = os.path.join(path, val_file)
    tst_path = os.path.join(path, tst_file)

    trn_files = read_file_list(trn_path)
    val_files = read_file_list(val_path)
    tst_files = read_file_list(tst_path)

    return trn_files, val_files, tst_files


def read_file_list(file_path):
    """
    Reads a text file containing a list of file names (one per line) and returns a Python list.

    :param file_path: path to the file containing the list of file names
    :return: list of file names
    """
    with open(file_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]

    return file_names


def increase_resolution(input_file, output_file):

    # Load annotations
    with open(input_file, 'r') as f:
        ann = json.load(f)

    # Update image resolution
    for i in range(len(ann["images"])):
        ann["images"][i]["width"] *= 2
        ann["images"][i]["height"] *= 2

    # Update bounding boxes
    width = ann["images"][0]["width"]
    height = ann["images"][0]["height"]
    for a in range(len(ann["annotations"])):
        for i in range(4):
            ann["annotations"][a]["bbox"][i] *= 2
        # Tiny (0.2% of a pixel)  decrease in width to avoid x + bw > iw (float precision)
        ann["annotations"][a]["bbox"][2] -= 0.002

        # print(img_id)
        bbox = ann["annotations"][a]["bbox"]
        x_max = bbox[0] + bbox[2]
        y_max = bbox[1] + bbox[3]
        if(x_max >= width):
            print(f"X max: {x_max}   width: {width}")
        if(y_max >= height):
            print(f"Y max: {y_max}   height: {height}")

    # Store the file
    with open(output_file, 'w') as f:
        json.dump(ann, f, indent=4)


def get_image(annotation, image_token):
    """
    Searches for an image on an annotation file
    """

    found_image = None
    for image in annotation['images']:
        # Check if the image name matches
        if image_token == get_file_id(image['file_name']):
            found_image = image
            break

    assert found_image is not None, f"Image {image_token} not found"
    return found_image

def get_annotation(annotation, image_id):
    """
    Searches for all annotations related to the provided image_id
    """
    found_annotations = []
    for ann in annotation["annotations"]:
        if ann["image_id"] == image_id:
            found_annotations.append(ann)
    assert len(found_annotations) > 0, "Annotations not found"
    return found_annotations

def create_subset_annotation(annotations, file_list, shuffle=True):
    """
    Creates a subset annotation file in the COCO format based on the provided list of filenames.

    Args:
        annotations (dict): The original annotation dictionary in COCO format.
        file_list (list): A list of filenames representing the subset.
        shuffle (bool, optional): Whether to shuffle the file list. Defaults to True. Not implemented yet.

    Returns:
        dict: The subset annotation dictionary in COCO format.

    Raises:
        AssertionError: If the provided file list length does not match the generated subset length.

    """
    # Copy fixed information
    subset_annotations = {
        'info': annotations['info'],
        'categories': annotations['categories'],
        'images': [],
        'annotations': []
    }
    # Update annotation category id from 0 to 1 (detectron2 prefers 0 for background)
    subset_annotations['categories'][0]['id'] = 1

    # Iterate through subset file_list
    for file_name in file_list:
        # print(f"DEBUG {file_name}")
        # Find corresponding iamge and annotations
        image = get_image(annotations, file_name)
        anns = get_annotation(annotations, image['id'])

        # Add image to new annotation
        # Fix image name
        subset_annotations['images'].append(image)
        # Add each annotation
        for ann in anns:
            # Fix category_id
            ann['category_id'] = 1
            subset_annotations['annotations'].append(ann)

    provided_len = len(file_list)
    generated_len = len(subset_annotations["images"])
    assert provided_len == generated_len , f"{provided_len} != {generated_len}"
    return subset_annotations

def generate_coco_split(annotation_file, trn_files, val_files, tst_files, tags, output_dir="./output"):
    """
    Splits a COCO format object detection annotations file into train, val, and test sets
    based on the provided lists of file names, and saves the split annotation files in the JSON format

    Args:
        annotation_file (str): The path to the original COCO format annotation file.
        trn_files (list): A list of training file names.
        val_files (list): A list of validation file names.
        tst_files (list): A list of testing file names.
        tags (list): A list of tags to append to the file names.
        output_dir (str, optional): The output directory path. Defaults to "./output".

    Returns:
        None

    """

    os.makedirs(output_dir, exist_ok=True)

    # Load the annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)


    for tag in tags:
        trn_annotations = create_subset_annotation(annotations, trn_files)
        for idx, image in enumerate(trn_annotations["images"]):
            new_name = get_file_id(image["file_name"])+tag+'.png'
            trn_annotations["images"][idx]["file_name"] = new_name

        val_annotations = create_subset_annotation(annotations, val_files)
        for idx, image in enumerate(val_annotations["images"]):
            new_name = get_file_id(image["file_name"])+tag+'.png'
            val_annotations["images"][idx]["file_name"] = new_name

        tst_annotations = create_subset_annotation(annotations, tst_files)
        for idx, image in enumerate(tst_annotations["images"]):
            new_name = get_file_id(image["file_name"])+tag+'.png'
            tst_annotations["images"][idx]["file_name"] = new_name

        # Save the split annotation files
        with open(os.path.join(output_dir, 'train'+tag+'.json'), 'w') as f:
            json.dump(trn_annotations, f, indent=4)
        with open(os.path.join(output_dir, 'val'+tag+'.json'), 'w') as f:
            json.dump(val_annotations, f, indent=4)
        with open(os.path.join(output_dir, 'test'+tag+'.json'), 'w') as f:
            json.dump(tst_annotations, f, indent=4)


def generate_single_json(annotation_file, file_list, tags, name, output_dir="./output"):
    """
    Creates COCO annotation file based on file list references.
    """
    output_dir = os.path.join(output_dir,'coco_splits')
    os.makedirs(output_dir, exist_ok=True)

    for tag in tags:
        # Load the annotations
        print(tag)
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        output = create_subset_annotation(annotations, file_list)
        for idx, image in enumerate(output["images"]):
            new_name = get_file_id(image["file_name"])+tag+'.png'
            output["images"][idx]["file_name"] = new_name

        with open(os.path.join(output_dir, name+tag+'.json'), 'w') as f:
            json.dump(output, f, indent=4)

def get_bbox_sizes(coco_ann):
    a_bbox_sizes = []
    s_bbox_sizes = []
    m_bbox_sizes = []
    l_bbox_sizes = []
    super_small = []

    for ann in coco_ann['annotations']:
        size = bbox = ann['bbox'][2] * ann['bbox'][3]
        a_bbox_sizes.append(size)

        if size <= (32**2):
            if size < (20):
                print("tiny bbox. Probably error")
                print(ann)
            elif size <= (8*8):
                super_small.append(size)
            s_bbox_sizes.append(size)
        elif size <= (96**2):
            m_bbox_sizes.append(size)
        else:
            l_bbox_sizes.append(size)
    print(f"Found {len(super_small)} super small bounding boxes")

    return a_bbox_sizes, s_bbox_sizes, m_bbox_sizes, l_bbox_sizes

def generate_yolo_split(trn_files, val_files, tst_files, tags, output_dir="./output"):
    """
    Generates YOLO format split text files for training, validation, and testing sets based on the provided lists of file names and tags.

    Args:
        trn_files (list): A list of training file names.
        val_files (list): A list of validation file names.
        tst_files (list): A list of testing file names.
        tags (list): A list of tags to append to the file names.
        output_dir (str, optional): The output directory path. Defaults to "./output".

    Returns:
        None

    """

    os.makedirs(output_dir, exist_ok=True)

    img_path = "./images/"

    for tag in tags:
        # Train files
        file_name = "train" + tag + ".txt"
        with open(os.path.join(output_dir, file_name), 'w') as f:
            for file_id in trn_files:
                f.write(img_path + file_id + tag + ".png")
                f.write("\n")

        # Validation files
        file_name = "val" + tag + ".txt"
        with open(os.path.join(output_dir, file_name), 'w') as f:
            for file_id in val_files:
                f.write(img_path + file_id + tag + ".png")
                f.write("\n")

        # Test files
        file_name = "test" + tag + ".txt"
        with open(os.path.join(output_dir, file_name), 'w') as f:
            for file_id in tst_files:
                f.write(img_path + file_id + tag + ".png")
                f.write("\n")

def create_wavy_flat_split():

    parser = argparse.ArgumentParser(description='Split the dataset')
    parser.add_argument('-f','--folder', help='Path to the dataset folder', default='../datasets/potato/labels/')
    args = parser.parse_args()

    wavy_files = read_file_list('input/wavy_test.txt')
    flat_files = read_file_list('input/flat_test.txt')

    annotation_file = "../datasets/potato/coco_splits/result.json"
    tags = ["_rgb","_rgb90","_rgbdif","_mono","_dolp","_pol"]
    generate_single_json(annotation_file, wavy_files, tags, "wavy", output_dir="./output")
    generate_single_json(annotation_file, flat_files, tags, "flat", output_dir="./output")


    print("Done")


def get_file_names(args):
    files = glob.glob(args.folder + "/*_rgb.txt")
    files.sort()
    return files

def large_bottles(label_path, min_size=790.0):
    with open(label_path) as topo_file:
        for line in topo_file:
            values = line.split()
            width = 1224.0 * float(values[3])
            height = 1024.0 * float(values[4])
            size = width * height
            if size < min_size:
                return False
    return True

def write_file_names_to_txt(selected_labels, output_file_name='./output/selected_files.txt'):

    with open(output_file_name, 'w') as f:
        for file_name in selected_labels:
            f.write(file_name + '\n')


def create_subset():

    parser = argparse.ArgumentParser(description='Split the dataset')
    parser.add_argument('-f','--folder', help='Path to the dataset folder', default='../datasets/potato/labels/')
    args = parser.parse_args()

    all_labels = get_file_names(args)

    selected_labels=[]
    for label_path in all_labels:
        if large_bottles(label_path):
            # name = os.path.basename((label_path)).split('.')[0]
            name = get_file_id(label_path)
            selected_labels.append(name)
    print("Selected images", len(selected_labels))

    write_file_names_to_txt(selected_labels)

    annotation_file = "../datasets/potato/coco_splits/result.json"
    tags = ["rgb"]
    generate_single_json(annotation_file, selected_labels, tags, "large_potatos", output_dir="./output")

    print("Done")


def count_bbox_per_image(coco_ann):
    bbox_per_image = {}
    for ann in coco_ann['annotations']:
        image_id = ann['image_id']
        if image_id not in bbox_per_image:
            bbox_per_image[image_id] = 0
        bbox_per_image[image_id] += 1

    return list(bbox_per_image.values())

def get_image_token(coco_ann, image_id):

    # Try direct access assuming images are sorted by id
    image_token = None
    if coco_ann["images"][image_id]["id"] == image_id:
        image_token = coco_ann["images"][image_id]["file_name"]
    # Linear search on the images
    else:
        for image in coco_ann["images"]:
            if image["id"] == image_id:
                image_token = image["file_name"]
                break
    # print(image_token)
    image_token = get_file_id(image_token)

    return image_token

def get_image_path(path="../datasets/potato/images/", token="exp01_frame00342_", ext="pol"):

    image_path = os.path.join(path, token+"_"+ext+".png")
    # TODO: test if image exists
    if os.path.exists(image_path) is False:
        print(f"ERROR: Could not find image {image_path}")

    return image_path


def crop_image(full_img, ann, size=100, show_label=False):
    height, width, _ = np.shape(full_img)
    x, y, w, h = ann["bbox"]

    cx = int(x + w/2)
    cy = int(y + h/2)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    xmin = max(cx - size//2, 0)
    xmax = min(cx + size//2, width-1)
    ymin = max(cy - size//2, 0)
    ymax = min(cy + size//2, height-1)

    # print(cx,cy)
    # print(xmin, xmax, ymin, ymax)

    if show_label:
        cv2.rectangle(full_img, (x,y), (x+w,y+h), (255,255,255), 1)

    cropped_img = full_img[ymin:ymax, xmin:xmax,:]
    return cropped_img

def save_images(boxes, img_token, exts):

    output_dir = "./output/crop_bbox"
    os.makedirs(output_dir, exist_ok=True)

    for index, ext in enumerate(exts):
        output_path = os.path.join(output_dir,img_token+"_"+ext+".png")
        cv2.imwrite(output_path, boxes[index])
        print("DEBUG", output_path)


def crop_bbox():
    exts = [ "mono", "rgb", "rgbdif", "dolp", "pol", "pauli"]
    ann_folder = "../datasets/potato/coco_splits/"
    ann_file = os.path.join(ann_folder, "result.json")
    size = 100

    with open(ann_file, 'r') as f:
        coco_ann = json.load(f)

    bbox_num = len(coco_ann["annotations"])
    print(f"BBOX num:{bbox_num}")

    random.shuffle(coco_ann["annotations"])

    for ann in coco_ann["annotations"]:
        print(ann["id"])
        img_token = get_image_token(coco_ann, ann["image_id"])

        tile = None
        boxes = []
        img_path = None
        for index, ext in enumerate(exts):
            img_path = get_image_path(token=img_token, ext=ext)
            full_img = cv2.imread(img_path)
            cropped_img = crop_image(full_img, ann, size)
            boxes.append(cropped_img)

        top_tile = cv2.hconcat(boxes[:3])
        bot_tile = cv2.hconcat(boxes[3:])
        tile = cv2.vconcat([top_tile, bot_tile])
        h, w, _ = np.shape(tile)
        tile = cv2.resize(tile, (w*4, h*4), interpolation = cv2.INTER_NEAREST)

        print("DEBUG",np.shape(tile))
        cv2.imshow('image',tile)

        k = cv2.waitKey(0) & 0xFF
        if k==27:    # Esc key to stop
            break
        elif k==ord('s'):
            save_images(boxes, img_token, exts)
        else:
            continue


def fix_yolo_detection(ann_file, det_file):
    """
    Fixes YOLO detections format according to provided annotation file.

    This function takes an annotation JSON file and a detection JSON file as input.
    It performs the following steps:
    1. Opens the annotation and detection JSON files.
    2. Processes the annotation file to create a dictionary mapping from file names to image IDs.
    3. Iterates through the detection file, updates the image ID using the dictionary map, and fixing the category ID.
    4. Stores the updated detection in a new JSON file.

    Args:
        ann_file (str): The path to the annotation JSON file.
        det_file (str): The path to the detection JSON file.

    Returns:
        None
    """

    # Check and open the annotatios and detection JSON files.
    with open(ann_file, 'r') as f:
        coco_ann = json.load(f)

    with open(det_file, 'r') as f:
        yolo_det = json.load(f)
        print(f"Input:\t{det_file}")


    # Process the ann_file to get a dictionary that maps from name to image id
    id_map = {}
    for image in coco_ann["images"]:
        name_tag = get_file_id(image["file_name"])
        img_id = image["id"]
        id_map[name_tag] = img_id
        # print(name_tag, img_id)

    # Iterate through the detection file and update the image id according to the dictionary map
    for idx, det in enumerate(yolo_det):
        old_id = get_file_id(det["image_id"])
        new_id = id_map[old_id]

        # Update value
        yolo_det[idx]["image_id"] = new_id
        # Also fix the category id
        yolo_det[idx]["category_id"] = 1
        # print(old_id, "->", new_id)

    # Store a new detection in the JSON file.
    directory, filename = os.path.split(det_file)
    name, extension = os.path.splitext(filename)
    new_name = name + "_fixed" + extension
    new_path = os.path.join(directory, new_name)
    print(f"Output:\t{new_path}")
    with open(new_path, 'w') as f:
            json.dump(yolo_det, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Split the dataset')
    parser.add_argument('-f','--folder', help='Path to the dataset folder', default='../datasets/potato/')
    args = parser.parse_args()

    return args
