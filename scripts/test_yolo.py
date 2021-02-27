import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from vinbigdata.scaled_yolov4.test import evaluate
from vinbigdata.scaled_yolov4.utils.datasets import create_dataloader
from vinbigdata.scaled_yolov4.utils.general import (ap_per_class, box_iou, check_file, check_img_size, clip_coords,
                                                    compute_loss, non_max_suppression, output_to_target, plot_images,
                                                    scale_coords, xywh2xyxy, xyxy2xywh)
from vinbigdata.scaled_yolov4.utils.torch_utils import select_device, time_synchronized

sys.path.append('vinbigdata/scaled_yolov4')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='configs/yolo/data.yaml', help='*.data path')
    parser.add_argument('--result-file', type=str, default='results/result.json', help='result file path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    if opt.task in ['val', 'test']:  # run normally
        evaluate(
            data=opt.data,
            task=opt.task,
            weights=opt.weights,
            batch_size=opt.batch_size,
            imgsz=opt.img_size,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            save_json=opt.save_json,
            augment=opt.augment,
            verbose=opt.verbose,
            merge=opt.merge,
            result_file=opt.result_file)
