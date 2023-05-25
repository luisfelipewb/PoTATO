import os
import argparse
import json
import numpy as np

# This is an implementation of SFOD, decribed in the paper "Simple Fusion of Object Detectors 
# for Improved Performance and Faster Deployment" It is a method for late fusion of detections from
# idependent models. In this implementation we assume the predictions are provided in a json coco format
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9359740

def parse_args():
    parser = argparse.ArgumentParser(description='SFOD implementation')
    parser.add_argument('-f','--folder', help='Path to the folder containing detections', default='./input')
    parser.add_argument('-c','--cs_thresh', help='Confidence Score Threshold', default=0.5, type=float)
    parser.add_argument('-n','--nms_thresh', help='IoU Threshold for NMS', default=0.8, type=float)
    parser.add_argument('-s','--sigma', help='Sigma value for updating score', default=0.3, type=float)

    args = parser.parse_args()

    return args

def load_detection_files(files, args):
    """
    Load JSON annotation files.

    Args:
    files (list): A list of file names to load.
    args (Namespace): The command-line arguments containing the folder

    Returns:
    list: A list of parsed JSON detections from the specified files.
    """

    annotations = []

    for ann_file in files:
        ann_path = os.path.join(args.folder, ann_file+'.json')
        with open(ann_path, 'r') as f:
            annotations.append(json.load(f))

    for idx, ann_file in enumerate(files):
        num_det = len(annotations[idx])
        # print(f"Loaded file {ann_file} with {num_det} detections ")

    return annotations

def combine_and_filter(all_detections, args):
    combined = []
    ignored = 0
    for detections in all_detections:
        for detection in detections:
            if detection["score"] > args.cs_thresh:
                # detection["score"] = detection["score"]/2
                combined.append(detection)
            else:
                ignored += 1

    combined.sort(key=lambda d: d['score'], reverse=True)
    combined.sort(key=lambda d: d['image_id'])

    # print(f"Combined detections: {len(combined)}. Ignored {ignored} dections bellow {args.cs_thresh} cs_thresh")
    return combined

def group_by_image(filtered, args):

    sorted_detections = []

    sorted_detections.append([filtered[0]])
    prev_image_id = filtered[0]["image_id"]
    # print(sorted_detections[-1][-1])

    for det in filtered[1:]:
        if prev_image_id == det["image_id"]:
            # Make sure detections on each image are sorted by the confidence
            assert(sorted_detections[-1][-1]["score"] >= det["score"])
            sorted_detections[-1].append(det)
            # print(sorted_detections[-1][-1])
        else:
            # print(" ")
            sorted_detections.append([det])
            prev_image_id = det["image_id"]
            # print(sorted_detections[-1][-1])

    return sorted_detections

def get_iou(bbox1, bbox2):


    (x1, y1, w1, h1), (x2, y2, w2, h2) = bbox1, bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    # Intersection area
    inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)),0) * \
            max((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)),0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter

    # IoU
    iou = inter / union
    # print(f"iou {iou}")
    return iou

def group_overlapping(images, args):

    overlapping = []

    for image in images:
        detections = image.copy()
        while(len(detections) > 0):
            group = []
            first = detections.pop(0)
            group.append(first)

            idx = 0
            while(idx < len(detections)):
                iou = get_iou(first["bbox"], detections[idx]["bbox"])
                if iou > args.nms_thresh:
                    group.append(detections.pop(idx))
                else:
                    idx +=1
            overlapping.append(group)

    return overlapping

def nms_filter(filtered, args):

    grouped = group_by_image(filtered, args)
    # print(f"Number of distinct images: {len(grouped)}")
    # print("Grouped by image_id")
    # for group in grouped:
    #     for detection in group:
    #         print(f"\t{detection}")
    #     print("")

    overlapping = group_overlapping(grouped, args)
    # print("Grouped by overlapping bboxes")
    # for group in overlapping:
    #     for detection in group:
    #         print(f"\t{detection}")
    #     print("")

    return overlapping

def rescore(group, sigma):

    score = group[0]["score"]
    for detection in group[1:]:
        score = 1 - (1-score) * pow(1-detection["score"], sigma)

    # score = 0
    # for detection in group:
    #     score += detection["score"]
    # socre = score / 6.0

    return score


def fuse_bbox(group):

    if len(group) == 1:
        return group[0]["bbox"]

    bbox = np.array([0.0,0.0,0.0,0.0])
    score = 0.0
    for detection in group:
        bbox += np.array(detection["bbox"]) * detection["score"]
        score += detection["score"]
    bbox = bbox / score

    return bbox.tolist()

def update_box_and_score(nms_filtered, args):
    updated_detections = []

    for group in nms_filtered:
        updated_detection = group[0].copy()
        updated_detection["score"] = rescore(group, args.sigma)
        updated_detection["bbox"] = fuse_bbox(group)
        updated_detections.append(updated_detection)

    # for idx, group in enumerate(nms_filtered):
    #     print("group")
    #     for det in group:
    #         print(f"  {det}")
    #     print("combined")
    #     print(f"  {updated_detections[idx]}")
    #     print("---")
    print(f"Final Number of detections: {len(updated_detections)}")

    return updated_detections

if __name__ == '__main__':

    # Get parameters including cs_thresh, nms_thresh, and sigma.
    args = parse_args()

    # Open the files for all detections
    files = ["dolp_small", "mono_small", ]
    files = ["dolp_small", "mono_small", "pol_small", "rgb90_small", "rgbdif_small", "rgb_small"]
    files = ["dolp", "mono", "pol", "rgb90", "rgbdif", "rgb"]
    all_detections = load_detection_files(files, args)
    

    cnt = 0

    for cs in np.arange(0.45, 0.65, 0.01):
        args.cs_thresh = cs
        for nms in np.arange(0.45, 0.6, 0.01):
            args.nms_thresh = nms

            cnt+=1
            folder = f'./output/merged/res{cnt:03d}'
            print(folder)
            os.makedirs(folder, exist_ok=True)

            for sigma in np.arange(0.1, 0.8, 0.05):
                args.sigma = sigma

                # Combiner + Confidence Score Filter
                filtered = combine_and_filter(all_detections, args)
                # print(filtered)
                # print(np.shape(filtered))

                # Inter-Detector NMS Filter
                nms_filtered = nms_filter(filtered, args)
                # print(np.shape(nms_filtered))635.17529296875

                #  Matched Detection Re-scorer and Bounding Box Fusion
                output = update_box_and_score(nms_filtered, args)

                # Store results
                test_name = f"cs{int(cs*100)}_nms{int(nms*100)}_s{int(sigma*100)}"
                output_path = os.path.join(folder, test_name+'.json')
                with open(output_path, 'w') as f:
                    json.dump(output, f)



