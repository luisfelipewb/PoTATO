from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np

import os

def draw_bounding_box(image, bbox, label):
    """
    Draws a bounding box with a label on the given image.
    
    Args:
        image (numpy.ndarray): The image on which to draw the bounding box.
        bbox (tuple): The bounding box coordinates in the format (x, y, width, height).
        label (str): The label to be displayed with the bounding box.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

    x, y, w, h = bbox

    
    color = (0,0,0)
    if label == 'gt':
        color = (0, 255, 0)  # green
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
    elif label == 'rgb' or label == 'dif' or label == 'mono' or label == 'pauli':
        color = (0, 0, 255)   # red
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
        cv2.rectangle(image, (x+w-tw, y-th), (x+w, y), color, cv2.FILLED) # Label text background
        cv2.putText(image, label, (x+w-tw, y), font, font_scale, (255, 255, 255), 1)  # Add the label
    elif label == 'pol' or label == 'dolp':
        color = (255, 255, 255)   # white
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
        cv2.rectangle(image, (x, y-th), (x+tw, y), color, cv2.FILLED) # Label text background
        cv2.putText(image, label, (x, y), font, font_scale, (0, 0, 0), 1)  # Add the label
    else: 
        color = (255, 255, 0) #
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw the bounding box
        cv2.rectangle(image, (x, y-th), (x+tw, y), color, cv2.FILLED) # Label text background
        cv2.putText(image, label, (x, y), font, font_scale, (0, 0, 0), 1)  # Add the label
        print("unknown label")

    

    return image


def show_detections(ground_truth, det_rgb, det_pol, img_id, img_dir = '../datasets/potato/images/'):
    """
    Show ground truth and detections for a given image ID.
    """
    # Load the image
    rgb_img_info = ground_truth.imgs[img_id]
    rgb_img_name = rgb_img_info['file_name']
    pol_img_name = rgb_img_name.replace('_rgb', '_pol')
    
    img_path = '{}/{}'.format(img_dir, rgb_img_name)
    img_rgb = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, pol_img_name)
    img_pol = cv2.imread(img_path)

    score_thresh = 0.75

    # Get the ground truth and detection bounding boxes
    gt_bbox = []
    for ann in ground_truth.imgToAnns[img_id]:
        # bbox = int(ann['bbox'])
        bbox = list(map(int,ann['bbox']))
        # bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        gt_bbox.append(bbox)

    det_rgb_bbox = []
    for ann in det_rgb.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            # bbox = int(ann['bbox'])
            bbox = list(map(int,ann['bbox']))
            # bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            det_rgb_bbox.append(bbox)

    det_pol_bbox = []
    for ann in det_pol.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            # bbox = int(ann['bbox'])
            bbox = list(map(int,ann['bbox']))
            # bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            det_pol_bbox.append(bbox)

    # det_dif_bbox = []


    # Draw bounding boxes
    color_gt = (0, 255, 0)  # green
    color_rgb = (0, 0, 255)   # red
    color_pol = (255, 0, 0)   # blue

    for bbox in gt_bbox:
        draw_bounding_box(img_rgb, bbox, 'gt')
        draw_bounding_box(img_pol, bbox, 'gt')

    for bbox in det_rgb_bbox:
        draw_bounding_box(img_rgb, bbox, 'rgb')

    for bbox in det_pol_bbox:
        draw_bounding_box(img_pol, bbox, 'pol')


    # cv2.putText(img_mono3, "MONO", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_rgb, "RGB", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    # cv2.putText(img_rgb_dif, "DIF", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    # cv2.putText(img_dolp_mono, "DOLP", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    cv2.putText(img_pol, "POL", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    # cv2.putText(img_pauli, "PAULI", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)

    tile = cv2.hconcat([img_rgb, img_pol])
    # cv2.imwrite(os.path.join(dir, token+"_stile.jpg"), stile)

    # # Large tile (6 images)
    # top_tile = cv2.hconcat([img_mono3, img_rgb, img_rgb_dif])
    # bot_tile = cv2.hconcat([img_dolp_mono, img_pol_mono, img_pauli])
    # tile = cv2.vconcat([top_tile, bot_tile])

    token = rgb_img_name.replace('_rgb.png', '')
    cv2.imwrite(os.path.join('./output/detections/', token+"_tile.jpg"), tile)

    # Show image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # output_path = './output/detections/' + rgb_img_name
    # cv2.imwrite(output_path, img_rgb)

    # output_path = './output/detections/' + pol_img_name
    # cv2.imwrite(output_path, img_pol)


def show_all_detections(ground_truth, detections, img_id, img_dir = '../datasets/potato/images/'):
    """
    Show ground truth and detections for a given image ID.
    """

    det_mono, det_rgb, det_dif, det_dolp, det_pol, det_pauli = detections

    # Load the image
    rgb_img_info = ground_truth.imgs[img_id]
    rgb_img_name = rgb_img_info['file_name']
    mono_img_name = rgb_img_name.replace('_rgb', '_mono')
    dif_img_name = rgb_img_name.replace('_rgb', '_rgbdif')
    dolp_img_name = rgb_img_name.replace('_rgb', '_dolp')
    pol_img_name = rgb_img_name.replace('_rgb', '_pol')
    pauli_img_name = rgb_img_name.replace('_rgb', '_pauli')

    
    img_path = '{}/{}'.format(img_dir, rgb_img_name)
    img_rgb = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, pol_img_name)
    img_pol = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, mono_img_name)
    img_mono = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, dif_img_name)
    img_dif = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, dolp_img_name)
    img_dolp = cv2.imread(img_path)
    img_path = '{}/{}'.format(img_dir, pauli_img_name)
    img_pauli = cv2.imread(img_path)


    score_thresh = 0.75

    # Get the ground truth and detection bounding boxes
    gt_bbox = []
    for ann in ground_truth.imgToAnns[img_id]:
        bbox = list(map(int,ann['bbox']))
        gt_bbox.append(bbox)

    det_rgb_bbox = []
    for ann in det_rgb.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_rgb_bbox.append(bbox)

    det_pol_bbox = []
    for ann in det_pol.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_pol_bbox.append(bbox)

    det_mono_bbox = []
    for ann in det_mono.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_mono_bbox.append(bbox)

    det_dif_bbox = []
    for ann in det_dif.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_dif_bbox.append(bbox)

    det_dolp_bbox = []
    for ann in det_dolp.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_dolp_bbox.append(bbox)

    det_pauli_bbox = []
    for ann in det_pauli.imgToAnns[img_id]:
        if ann['score'] > score_thresh:
            bbox = list(map(int,ann['bbox']))
            det_pauli_bbox.append(bbox)


    # Draw bounding boxes
    color_gt = (0, 255, 0)  # green
    color_rgb = (0, 0, 255)   # red
    color_pol = (255, 0, 0)   # blue

    for bbox in gt_bbox:
        draw_bounding_box(img_mono, bbox, 'gt')
        draw_bounding_box(img_rgb, bbox, 'gt')
        draw_bounding_box(img_dif, bbox, 'gt')
        draw_bounding_box(img_dolp, bbox, 'gt')
        draw_bounding_box(img_pol, bbox, 'gt')
        draw_bounding_box(img_pauli, bbox, 'gt')

    for bbox in det_rgb_bbox:
        draw_bounding_box(img_rgb, bbox, 'rgb')

    for bbox in det_pol_bbox:
        draw_bounding_box(img_pol, bbox, 'pol')

    for bbox in det_mono_bbox:
        draw_bounding_box(img_mono, bbox, 'mono')

    for bbox in det_dif_bbox:
        draw_bounding_box(img_dif, bbox, 'dif')
    
    for bbox in det_dolp_bbox:
        draw_bounding_box(img_dolp, bbox, 'dolp')
    
    for bbox in det_pauli_bbox:
        draw_bounding_box(img_pauli, bbox, 'pauli')
    


    cv2.putText(img_mono, "MONO", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_rgb, "RGB", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_dif, "DIF", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
    cv2.putText(img_dolp, "DOLP", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    cv2.putText(img_pol, "POL", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
    cv2.putText(img_pauli, "PAULI", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)

    tile = cv2.hconcat([img_rgb, img_pol])
    # cv2.imwrite(os.path.join(dir, token+"_stile.jpg"), stile)

    # # Large tile (6 images)
    top_tile = cv2.hconcat([img_mono, img_rgb, img_dif])
    bot_tile = cv2.hconcat([img_dolp, img_pol, img_pauli])
    tile = cv2.vconcat([top_tile, bot_tile])

    token = rgb_img_name.replace('_rgb.png', '')
    cv2.imwrite(os.path.join('./output/detections/', token+"_tile.jpg"), tile)

    # Show image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # output_path = './output/detections/' + rgb_img_name
    # cv2.imwrite(output_path, img_rgb)

    # output_path = './output/detections/' + pol_img_name
    # cv2.imwrite(output_path, img_pol)

if __name__ == "__main__":

    coco_gt = COCO(annotation_file='../datasets/potato/split_seq/test_rgb.json')
    coco_det_mono = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_mono/test/coco_instances_results.json")
    coco_det_rgb = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_rgb/test/coco_instances_results.json")
    coco_det_dif = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_rgbdif/test/coco_instances_results.json")
    coco_det_dolp = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_dolp/test/coco_instances_results.json")
    coco_det_pol = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_pol/test/coco_instances_results.json")
    coco_det_pauli = coco_gt.loadRes("../experiments/runs/det2_100/fasterrcnn_pauli/test/coco_instances_results.json")

    coco_eval_mono = COCOeval(coco_gt, coco_det_mono, 'bbox')
    coco_eval_rgb = COCOeval(coco_gt, coco_det_rgb, 'bbox')
    coco_eval_dif = COCOeval(coco_gt, coco_det_dif, 'bbox')
    coco_eval_dolp = COCOeval(coco_gt, coco_det_dolp, 'bbox')
    coco_eval_pol = COCOeval(coco_gt, coco_det_pol, 'bbox')
    coco_eval_pauli = COCOeval(coco_gt, coco_det_pauli, 'bbox')

    coco_eval_mono.evaluate()
    coco_eval_rgb.evaluate()
    coco_eval_dif.evaluate()
    coco_eval_dolp.evaluate()
    coco_eval_pol.evaluate()
    coco_eval_pauli.evaluate()

    detections = [coco_det_mono, coco_det_rgb, coco_det_dif, coco_det_dolp, coco_det_pol, coco_det_pauli]
    # iterate over all images in the ground truth and print the file names
    for img_id in coco_gt.imgs:
        img_info = coco_gt.imgs[img_id]
        print(img_info["file_name"])
        show_all_detections(coco_gt, detections, img_id)

        
        # quit()



    # print("hello")
    # print(len(coco_eval_rgb.ious))
    # print(coco_eval_rgb.ious.keys())
    # for key in coco_eval_rgb.ious:
    #     print(key)
    #     # ious_rgb = coco_eval_rgb.ious[key] 
        # ious_pol = coco_eval_pol.ious[key]


    # for key in sorted_dict[-10:]:
    #     img_info = coco_gt.imgs[key[0]]
    #     print(img_info["file_name"])

    #     print(key[0])
    #     print("rgb", coco_eval_rgb.ious[key])
    #     print("pol", coco_eval_pol.ious[key])

    #     print()

        # show_detections(coco_gt, coco_det_rgb, coco_det_pol, key[0])
        




quit()
idx = 2

print("evalImgs")
print('dtMatches')
print(coco_eval_rgb.evalImgs[idx]['dtMatches'])
print('gtMatches')
print(coco_eval_rgb.evalImgs[idx]['gtMatches'])
print("")
print(coco_eval_rgb.evalImgs[idx])
print("\n\n\n")

coco_eval_rgb.accumulate()
coco_eval_pol.accumulate()
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.


# coco_eval_rgb.summarize()
# coco_eval_pol.summarize()



# quit()

max_difs = {}
# maxd = 0
# print(cocoEval.evalImgs[1])
# print(cocoEval.ious.keys())
for key in coco_eval_rgb.ious:


    ious_rgb = coco_eval_rgb.ious[key] 
    ious_pol = coco_eval_pol.ious[key]

    print("rgb ious")
    print(ious_rgb)
    print("pol ious")
    print(ious_pol)

    max_rgb = np.max(ious_rgb, axis=0)
    max_pol = np.max(ious_pol, axis=0)
    max_dif = np.max(max_pol - max_rgb)

    max_difs[key] = max_dif

    # if max_dif > maxd:
    #     print(key)
    #     print("max")
    #     print("rgb", max_rgb)
    #     print("pol", max_pol)
    #     print(f"max diff {max_dif}")
    #     maxd = max_dif
    #     print(max_difs[key])


sorted_dict = sorted(max_difs, key=max_difs.get)




# for key in cocoEval.ious.keys():
#     print()
#     print(cocoEval.ious[key])

# print()
# Disctionary of images in the ground truth
# for img_id in cocoAnnotation.imgs:
#     print(img_id)

# img_id = 4677
# print(coco_gt.getAnnIds(imgIds=img_id))

# ann_ids = coco_gt.getAnnIds(imgIds=img_id)
# gt_anns = coco_gt.loadAnns(ann_ids)

# print(gt_anns)

