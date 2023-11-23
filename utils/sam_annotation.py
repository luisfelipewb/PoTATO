import os
import cv2
import numpy as np
from ultralytics.models.sam import Predictor as SAMPredictor

overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="sam_b.pt", save=False)
predictor = SAMPredictor(overrides=overrides)

label_dir = "../datasets/potato/labels/"
filename_base_path = "../datasets/potato/"
data_dir = "../datasets/potato/images/"

filename_list = ["test.txt", "train.txt", "val.txt"]



image_width = 1224
image_height = 1024

def get_bboxes(file):
    lines = file.readlines()

    bboxes = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x_min = int((center_x - width/2) * image_width)
        y_min = int((center_y - height/2) * image_height)
        x_max = int((center_x + width/2) * image_width)
        y_max = int((center_y + height/2) * image_height)

        bbox = [x_min, y_min, x_max, y_max]
        bboxes.append(bbox)
    return bboxes


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def get_np_mask(results):
    """Converts the mask tensor to a numpy array.
    Params:
        results: The results of the model inference.
    Returns:
        merged_masks: The merged masks. np.ndarray (Boolen)
    """
    # Convert tensor to numpy array
    masks_np = results[0].masks.cpu().numpy()

    # Merge the masks into a single category
    merged_masks = np.any(masks_np.data, axis=0)

    return merged_masks


count = 0

for filename in filename_list:
    print(filename)
    filename_path = os.path.join(filename_base_path, filename)
    with open(filename_path, 'r') as file:
        labels = file.readlines()

    for label in labels:

        rgb_label_path = os.path.join(label_dir, label.replace("\n", "_rgb.txt"))
        rgb_image_path = os.path.join(data_dir, label.replace("\n", "_rgb.png"))

        pol_label_path = os.path.join(label_dir, label.replace("\n", "_pol.txt"))
        pol_image_path = os.path.join(data_dir, label.replace("\n", "_pol.png"))

        with open(rgb_label_path, 'r') as file:
            rgb_bboxes = get_bboxes(file)
        with open(pol_label_path, 'r') as file:
            pol_bboxes = get_bboxes(file)

        predictor.set_image(rgb_image_path)
        rgb_results = predictor(bboxes=rgb_bboxes, labels=1, save_img=False)
        predictor.reset_image()

        predictor.set_image(pol_image_path)
        pol_results = predictor(bboxes=pol_bboxes, labels=1, save_img=False)
        predictor.reset_image()

        rgb_mask = get_np_mask(rgb_results)
        pol_mask = get_np_mask(pol_results)
        combined_mask = np.logical_or(rgb_mask, pol_mask)

        rgb_mask_img = np.where(rgb_mask, 255, 0).astype(np.uint8)
        pol_mask_img = np.where(pol_mask, 255, 0).astype(np.uint8)
        combined_mask_img = np.where(combined_mask, 255, 0).astype(np.uint8)

        #Save the images
        # cv2.imwrite(os.path.join("./runs/masks", label.replace("\n", "_rgb_mask.png")), rgb_mask_img)
        # cv2.imwrite(os.path.join("./runs/masks", label.replace("\n", "_pol_mask.png")), pol_mask_img)
        cv2.imwrite(os.path.join("./runs/masks", label.replace("\n", "_mask.png")), combined_mask_img)

        # count += 1
        if count > 0:
            break

quit()
