import os
import numpy as np
import argparse

from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances


def parse_arguments():
  parser = argparse.ArgumentParser(description='Parameters for comparing the models')
  parser.add_argument('-c','--config', help='.Configuration file (.yaml) ',required=True)
  parser.add_argument('-t','--tag', help='Tag identifying the image channel [mono, rgb, rgb90, rgbdif, dolp, pol]', required=True)
  parser.add_argument('-d','--device', help='Device that should be used for training. eg. cuda:0', default="cuda:1")
  parser.add_argument('-w','--weights', help='Weights file')
  parser.add_argument('-o','--output', help='output directory', default="./runs/")
  parser.add_argument('-a', '--ann_folder', help='Folder containing the annotations')
  parser.add_argument('-i', '--img_folder', help='Folder containing the images')
  parser.add_argument('--basename', default="basename")
  parser.add_argument('--eval_only', action='store_true')

  args = parser.parse_args()
  args.tag = '_'+args.tag
  args.basename = os.path.basename(args.config).split('.')[0]+args.tag
  args.output = os.path.join(args.output, args.basename)
  return args

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")
            if(os.path.exists(output_folder)):
              output_folder = os.path.join(cfg.OUTPUT_DIR, "test")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def register_datasets(args):

  ann_folder = args.ann_folder
  img_folder = args.img_folder

  train_dataset = 'train'+args.tag
  val_dataset = 'val'+args.tag

  train_ann_path = os.path.join(ann_folder, train_dataset+'.json')
  val_ann_path = os.path.join(ann_folder, val_dataset+'.json')

  register_coco_instances(train_dataset, {}, train_ann_path, img_folder )
  register_coco_instances(val_dataset, {}, val_ann_path, img_folder)

  test_dataset = 'Empty'  
  if args.tag != '_flow':
    test_dataset = 'test'+args.tag
    test_ann_path = os.path.join(ann_folder, test_dataset+'.json')
    register_coco_instances(test_dataset, {}, test_ann_path, img_folder)

  return train_dataset, val_dataset, test_dataset

def update_config(args, train_dataset, test_dataset, test=False):
  cfg = get_cfg()
  
  cfg.merge_from_file(args.config)
  cfg.merge_from_file("./config/common.yaml")

  cfg.DATASETS.TRAIN = (train_dataset, )
  cfg.DATASETS.TEST = (test_dataset, )
  cfg.MODEL.DEVICE = args.device
  cfg.OUTPUT_DIR = args.output

  # Save configuration file
  config_file_name = "config_"
  if test:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    config_file_name = config_file_name + "test.yaml"
  else:
    config_file_name = config_file_name + "train.yaml"

  with open(os.path.join(args.output, config_file_name), 'w') as f:
    f.write(str(cfg))

  return cfg



if __name__ == "__main__": 
  args = parse_arguments()
  os.makedirs(args.output, exist_ok=True)

  setup_logger()

  train_dataset, val_dataset, test_dataset = register_datasets(args)

  cfg = update_config(args, train_dataset, val_dataset)


  if args.eval_only:
    print("EVALUATION")
    if test_dataset == 'Empty':
      cfg = update_config(args, train_dataset, val_dataset, test=True)
    else:
      cfg = update_config(args, train_dataset, test_dataset, test=True)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    res = Trainer.test(cfg, model)
  else:
    print("TRAINING")
    cfg = update_config(args, train_dataset, val_dataset)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
