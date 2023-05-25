from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv
import yaml
from yaml.loader import SafeLoader


def load_all_experiments(path='./input/experiments.yaml'):
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

    cocoEval.params.maxDets=[1, 5, 10]
    cocoEval.params.useCats = 0
    # cocoEval.params.iouThrs = iou_thresholds
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[:]

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

    for gt, dt_dict in experiments:
        for item in dt_dict:
            name = item["name"]
            dt = item["dt"]
            stats = evaluate(gt, dt)
            metrics[name] = stats

    return metrics


def generate_coco_metrics_csv(metrics, path='output/metrics.csv'):

    """
    Generate a CSV file that summarizes the evaluation metrics computed by the COCO evaluation script.

    Args:
        metrics (dict): a dictionary that maps experiment names to the evaluation statistics produced by the COCO
            evaluation script. The keys of the dictionary are the experiment names, and the values are lists of
            evaluation metrics (e.g., precision, recall, AP) computed by the COCO evaluation script for different
            object categories and different intersection-over-union thresholds.
        path (str): the path to the output CSV file.

    Returns:
        None. The function writes the output CSV file to disk.

    The output CSV file contains one row for each experiment in the input `metrics` dictionary. The first column
    of each row contains the name of the experiment, and the remaining columns contain the evaluation metrics
    computed by the COCO evaluation script. The columns correspond to the following metrics and IoU thresholds:

    - AP: average precision
    - AP50: average precision with IoU threshold of 0.5
    - AP75: average precision with IoU threshold of 0.75
    - APs: average precision for small objects (area < 32^2 pixels)
    - APm: average precision for medium-sized objects (32^2 pixels <= area < 96^2 pixels)
    - APl: average precision for large objects (area >= 96^2 pixels)
    - AR1: average recall with 1 detection per image
    - AR5: average recall with 5 detections per image
    - AR10: average recall with 10 detections per image
    - ARs: average recall for small objects (area < 32^2 pixels)
    - ARm: average recall for medium-sized objects (32^2 pixels <= area < 96^2 pixels)
    - ARl: average recall for large objects (area >= 96^2 pixels)
    """

    with open(path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Generate header
        header = ["channel", "AP","AP50","AP75","APs","APm","APl","AR1","AR5","AR10","ARs","ARm","ARl"]
        writer.writerow(header)

        # Add row with metrics for each experiment
        for name in metrics:
            row = [name] + metrics[name].tolist()
            writer.writerow(row)


if __name__ == '__main__':
    experiments = load_all_experiments()
    metrics = run_evaluations(experiments)
    generate_coco_metrics_csv(metrics)
    print("Done")
