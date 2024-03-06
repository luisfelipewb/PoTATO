from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv
import yaml
from yaml.loader import SafeLoader
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600


def load_all_experiments(path='./experiments.yaml'):
    """
    Load ground-truth and detection results from a YAML file that specifies multiple experiments.

    Args:
        path (str, optional): path to the YAML file that contains experiment information.
            The file should have the following format:
            ```
            <path-to-ground-truth-annotations_1>
              <detection_name_1>: <path-to-detection-results-1>
              <detection_name_2>: <path-to-detection-results-2>
            <path-to-ground-truth-annotations_2>
              <detection_name_3>: <path-to-detection-results-3>
              <detection_name_4>: <path-to-detection-results-4>
            ```
            Note that multiple detections can share the same ground truth and that is only valid becase the 
            different experiments can be multiple channels of the same image, therefore the bounding boxes are
            still in the same place. The references to the images are still valid because by the ID number (and not the name) is used.
            Default is './input/experiments.yaml'.

    Returns:
        list: a list of tuples,
          where each tuple contains the ground-truth annotations (as a COCO object)
              and a list of detection results for a specific experiment.
              The list of detection results is represented as a list of dictionaries, where each dictionary
              has the keys 'name', 'path', and 'dt', which respectively specify the name of the experiment,
              the path to the detection results file, and the COCO object that represents the detection results.
    """
    with open(path, 'r') as f:

        data = list(yaml.load_all(f, Loader=SafeLoader))[0]
        print(data)

        experiments = []

        for gt_path in data:
            # print(f"gt_path: {gt_path}:")
            gt = COCO(gt_path)
            dts = []
            for exp_name in data[gt_path]:
                exp_path = data[gt_path][exp_name]
                print(f"  exp_name: {exp_name} exp_path: {exp_path}")
                dt = gt.loadRes(exp_path)

                dts.append({"name":exp_name, "path":exp_path, "dt":dt})

            experiments.append((gt,dts))

        return experiments


def evaluate(cocoGt, cocoDt):
    """
    Run COCO evaluation on object detection results.

    Args:
        cocoGt (COCO): an instance of COCO class that represents the ground-truth annotations.
        cocoDt (COCO): an instance of COCO class that represents the detection results.

    Returns:
        list: a list of performance metrics for different evaluation criteria and IoU thresholds.
              The order of the metrics in the list is as follows:
              [AP, AP50, AP75, APs, APm, APl, AR1, AR5, AR10, ARs, ARm, ARl].
              Here, AP means average precision, AR means average recall, and
              s, m, and l denote small, medium, and large objects, respectively.
              The numbers 1, 5, and 10 denote the maximum number of detections per image.
    """
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')

    #cocoEval.params.maxDets=[1, 5, 10]
    cocoEval.params.useCats = 0
    # cocoEval.params.iouThrs = iou_thresholds
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats = cocoEval.stats[:]

    # precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
    # recall      = -np.ones((T,  K,A,M))
    # T           = len(p.iouThrs)
    # R           = len(p.recThrs)
    # K           = len(p.catIds) if p.useCats else 1
    # A           = len(p.areaRng)
    # M           = len(p.maxDets)
    # [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95] <- IoU threshold
    # [0    1    2    3    4    5    6    7    8    9] <- index
    p_idx = 0
    precisions = {}
    precisions['all'] = cocoEval.eval['precision'][p_idx, :, 0, 0, 2]
    precisions['small'] = cocoEval.eval['precision'][p_idx, :, 0, 1, 2]
    precisions['medium'] = cocoEval.eval['precision'][p_idx, :, 0, 2, 2]
    precisions['large'] = cocoEval.eval['precision'][p_idx, :, 0, 3, 2]

    p_idx = 0
    precision_50 = {}
    precision_50['small'] = cocoEval.eval['precision'][p_idx, :, 0, 0, 2]
    precision_50['medium'] = cocoEval.eval['precision'][p_idx, :, 0, 2, 2]
    precision_50['large'] = cocoEval.eval['precision'][p_idx, :, 0, 3, 2]

    p_idx = 5
    precision_75 = {}
    precision_75['small'] = cocoEval.eval['precision'][p_idx, :, 0, 0, 2]
    precision_75['medium'] = cocoEval.eval['precision'][p_idx, :, 0, 2, 2]
    precision_75['large'] = cocoEval.eval['precision'][p_idx, :, 0, 3, 2]

    p_idx = 7
    precision_85 = {}
    precision_85['small'] = cocoEval.eval['precision'][p_idx, :, 0, 0, 2]
    precision_85['medium'] = cocoEval.eval['precision'][p_idx, :, 0, 2, 2]
    precision_85['large'] = cocoEval.eval['precision'][p_idx, :, 0, 3, 2]

    return stats, precisions, precision_50, precision_75, precision_85


def run_evaluations(experiments):
    """
    Run object detection evaluation for a list of experiments.

    Args:
        experiments (list): a list of tuples, where each tuple contains the ground-truth annotations (as a COCO object)
            and a list of detection results for a specific experiment.
            The list of detection results is represented as a list of dictionaries, where each dictionary
            has the keys 'name', 'path', and 'dt', which respectively specify the name of the experiment,
            the path to the detection results file, and the COCO object that represents the detection results.

    Returns:
        dict: a dictionary that maps experiment names to the evaluation statistics produced by the COCO evaluation script.
            The keys of the dictionary are the experiment names, and the values are lists of evaluation metrics
            (e.g., precision, recall, AP) computed by the COCO evaluation script for different object categories and
            different intersection-over-union thresholds.
    """

    metrics = {}
    precisions = {}
    precisions_50 = {}
    precisions_75 = {}
    precisions_85 = {}


    for gt, dt_dict in experiments:
        for item in dt_dict:
            name = item["name"]
            dt = item["dt"]
            stats, precision, precision50, precision75, precision85 = evaluate(gt, dt)
            metrics[name] = stats
            precisions[name] = precision
            precisions_50[name] = precision50
            precisions_75[name] = precision75
            precisions_85[name] = precision85



    return metrics, precisions, precisions_50, precisions_75, precisions_85


def plot_precision_recall_curve_all(precisions):
    """
    Plot the precision-recall curve for a list of experiments.
    """
    # iterate through each key in the dictionary and get index

    x = np.arange(0, 1.01, 0.01)
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    axs[0,0].set_title('Small Bounding Boxes')
    axs[0,1].set_title('Medium Bounding Boxes')
    axs[0,2].set_title('Large Bounding Boxes')
    axs[0,3].set_title('All Bounding Boxes')


    for key in precisions:
        print(key)
        # if key contains "yolo" then plot on first line
        if "yolo" in key:
            axs[0,0].plot(x, precisions[key]["small"], label=key)
            axs[0,1].plot(x, precisions[key]["medium"], label=key)
            axs[0,2].plot(x, precisions[key]["large"], label=key)
            axs[0,3].plot(x, precisions[key]["all"], label=key)
            
        if "faster" in key:
            axs[1,0].plot(x, precisions[key]["small"], label=key)
            axs[1,1].plot(x, precisions[key]["medium"], label=key)
            axs[1,2].plot(x, precisions[key]["large"], label=key)
            axs[1,3].plot(x, precisions[key]["all"], label=key)

        if "retina" in key:
            axs[2,0].plot(x, precisions[key]["small"], label=key)
            axs[2,1].plot(x, precisions[key]["medium"], label=key)
            axs[2,2].plot(x, precisions[key]["large"], label=key)
            axs[2,3].plot(x, precisions[key]["all"], label=key)
            

    axs[0,0].legend()
    axs[0,1].legend()
    axs[0,2].legend()
    axs[0,3].legend()
    axs[1,0].legend()
    axs[1,1].legend()
    axs[1,2].legend()
    axs[1,3].legend()
    axs[2,0].legend()
    axs[2,1].legend()
    axs[2,2].legend()
    axs[2,3].legend()

    axs[2,0].set_xlabel('Recall')
    axs[2,1].set_xlabel('Recall')
    axs[2,2].set_xlabel('Recall')
    axs[2,3].set_xlabel('Recall')
    axs[0,0].set_ylabel('Precision')
    axs[1,0].set_ylabel('Precision')
    axs[2,0].set_ylabel('Precision')
    

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # plt.plot(x, yolo_rgb[:,2], label='medium')
    # plt.plot(x, yolo_rgb[:,3], label='large')
    
    path='./output/precision_recall_curve_all.png'
    plt.savefig(path)
    
def plot_precision_recall_curve(precisions):
    """
    Plot the precision-recall curve for a list of experiments.
    """
    fasterrcnn_rgb = precisions["fasterrcnn_rgb"]
    fasterrcnn_rgbdif = precisions["fasterrcnn_rgbdif"]
    fasterrcnn_pol = precisions["fasterrcnn_pol"]

    # iterate through each key in the dictionary and get index

    x = np.arange(0, 1.01, 0.01)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].set_title('Small$(area < 32^2)$')
    axs[1].set_title('Medium$(32^2 < area < 96^2)$')
    axs[2].set_title('Large$(96^2 < area)$')
    axs[3].set_title('All')



    axs[0].plot(x, fasterrcnn_rgb["small"], label="RGB")
    axs[0].plot(x, fasterrcnn_pol["small"], label="POL")
    axs[0].plot(x, fasterrcnn_rgbdif["small"], label="RGBDIF")

    axs[1].plot(x, fasterrcnn_rgb["medium"], label="RGB")
    axs[1].plot(x, fasterrcnn_pol["medium"], label="POL")
    axs[1].plot(x, fasterrcnn_rgbdif["medium"], label="RGBDIF")

    axs[2].plot(x, fasterrcnn_rgb["large"], label="RGB")
    axs[2].plot(x, fasterrcnn_pol["large"], label="POL")
    axs[2].plot(x, fasterrcnn_rgbdif["large"], label="RGBDIF")

    axs[3].plot(x, fasterrcnn_rgb["all"], label="RGB")    
    axs[3].plot(x, fasterrcnn_pol["all"], label="POL")
    axs[3].plot(x, fasterrcnn_rgbdif["all"], label="RGBDIF")

    axs[0].legend()
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    
    axs[0].set_xlabel('Recall')
    axs[1].set_xlabel('Recall')
    axs[2].set_xlabel('Recall')
    axs[3].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')

    plt.tight_layout()

    path='./output/precision_recall_curve.png'
    plt.savefig(path)

def plot_precision_recall_paper(precisions_50, precisions_75, precisions_85):

    fasterrcnn_rgb_50 = precisions_50["fasterrcnn_rgb"]
    fasterrcnn_rgbdif_50 = precisions_50["fasterrcnn_rgbdif"]
    fasterrcnn_pol_50 = precisions_50["fasterrcnn_pol"]

    fasterrcnn_rgb_75 = precisions_75["fasterrcnn_rgb"]
    fasterrcnn_rgbdif_75 = precisions_75["fasterrcnn_rgbdif"]
    fasterrcnn_pol_75 = precisions_75["fasterrcnn_pol"]

    fasterrcnn_rgb_85 = precisions_85["fasterrcnn_rgb"]
    fasterrcnn_rgbdif_85 = precisions_85["fasterrcnn_rgbdif"]
    fasterrcnn_pol_85 = precisions_85["fasterrcnn_pol"]

    x = np.arange(0, 1.01, 0.01)
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))

    axs[0,0].plot(x, fasterrcnn_rgb_50["small"], label="RGB", linewidth=2, alpha=0.9)
    axs[0,0].plot(x, fasterrcnn_rgbdif_50["small"], label="DIF", linewidth=2, alpha=0.9)
    axs[0,0].plot(x, fasterrcnn_pol_50["small"], label="POL", linewidth=2, alpha=0.9)
    axs[0,1].plot(x, fasterrcnn_rgb_50["medium"], label="RGB", linewidth=2, alpha=0.9)
    axs[0,1].plot(x, fasterrcnn_rgbdif_50["medium"], label="DIF", linewidth=2, alpha=0.9)
    axs[0,1].plot(x, fasterrcnn_pol_50["medium"], label="POL ", linewidth=2, alpha=0.9)
    axs[0,2].plot(x, fasterrcnn_rgb_50["large"], label="RGB ", linewidth=2, alpha=0.9)
    axs[0,2].plot(x, fasterrcnn_rgbdif_50["large"], label="DIF ", linewidth=2, alpha=0.9)
    axs[0,2].plot(x, fasterrcnn_pol_50["large"], label="POL ", linewidth=2, alpha=0.9)
    axs[1,0].plot(x, fasterrcnn_rgb_75["small"], label="RGB", linewidth=2, alpha=0.9)
    axs[1,0].plot(x, fasterrcnn_rgbdif_75["small"], label="DIF", linewidth=2, alpha=0.9)
    axs[1,0].plot(x, fasterrcnn_pol_75["small"], label="POL", linewidth=2, alpha=0.9)
    axs[1,1].plot(x, fasterrcnn_rgb_75["medium"], label="RGB", linewidth=2, alpha=0.9)
    axs[1,1].plot(x, fasterrcnn_rgbdif_75["medium"], label="DIF", linewidth=2, alpha=0.9)
    axs[1,1].plot(x, fasterrcnn_pol_75["medium"], label="POL", linewidth=2, alpha=0.9)
    axs[1,2].plot(x, fasterrcnn_rgb_75["large"], label="RGB", linewidth=2, alpha=0.9)
    axs[1,2].plot(x, fasterrcnn_rgbdif_75["large"], label="DIF", linewidth=2, alpha=0.9)
    axs[1,2].plot(x, fasterrcnn_pol_75["large"], label="POL", linewidth=2, alpha=0.9)
    axs[2,0].plot(x, fasterrcnn_rgb_85["small"], label="RGB", linewidth=2, alpha=0.9)
    axs[2,0].plot(x, fasterrcnn_rgbdif_85["small"], label="DIF", linewidth=2, alpha=0.9)
    axs[2,0].plot(x, fasterrcnn_pol_85["small"], label="POL", linewidth=2, alpha=0.9)
    axs[2,1].plot(x, fasterrcnn_rgb_85["medium"], label="RGB", linewidth=2, alpha=0.9)
    axs[2,1].plot(x, fasterrcnn_rgbdif_85["medium"], label="DIF", linewidth=2, alpha=0.9)
    axs[2,1].plot(x, fasterrcnn_pol_85["medium"], label="POL", linewidth=2, alpha=0.9)
    axs[2,2].plot(x, fasterrcnn_rgb_85["large"], label="RGB", linewidth=2, alpha=0.9)
    axs[2,2].plot(x, fasterrcnn_rgbdif_85["large"], label="DIF", linewidth=2, alpha=0.9)
    axs[2,2].plot(x, fasterrcnn_pol_85["large"], label="POL", linewidth=2, alpha=0.9)

    axs[2,0].set_xlabel('Recall',)
    axs[2,1].set_xlabel('Recall',)
    axs[2,2].set_xlabel('Recall',)
    axs[0,0].set_ylabel('Precision',)
    axs[1,0].set_ylabel('Precision',)
    axs[2,0].set_ylabel('Precision',)

    line1 = axs[0,2].twinx()
    line2 = axs[1,2].twinx()
    line3 = axs[2,2].twinx()
    line1.set_ylabel('IoU = 0.50', fontsize="18", )
    line2.set_ylabel('IoU = 0.75', fontsize="18", )
    line3.set_ylabel('IoU = 0.85', fontsize="18", )

    line1.axes.yaxis.set_ticks([])
    line2.axes.yaxis.set_ticks([])
    line3.axes.yaxis.set_ticks([])
    axs[0,0].set_title('Small $(area < 32^2)$')
    axs[0,1].set_title('Medium $(32^2 < area < 96^2)$')
    axs[0,2].set_title('Large $(96^2 < area)$')
    
    axs[0,0].legend(fontsize="10")
    axs[0,1].legend(fontsize="10")
    axs[0,2].legend(fontsize="10")
    axs[1,0].legend(fontsize="10")
    axs[1,1].legend(fontsize="10")
    axs[1,2].legend(fontsize="10")
    axs[2,0].legend(fontsize="10")
    axs[2,1].legend(fontsize="10")
    axs[2,2].legend(fontsize="10")

    plt.tight_layout()
    path='./output/precision_recall_curve_paper.png'
    plt.savefig(path)


if __name__ == '__main__':
    experiments = load_all_experiments()
    metrics, precisions, precisions_50, precisions_75, precisions_85 = run_evaluations(experiments)

    plot_precision_recall_paper(precisions_50, precisions_75, precisions_85)


    print("Done")

