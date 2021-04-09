# %%
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from mmcv import Config
from vinbigdata.postprocess import nms_models, normal_by_boxes
from vinbigdata.testing import batch_inference, calc_measure, generate_gt_boxes, predict_boxes
from vinbigdata.utils import is_interactive, rel2abs
from vinbigdata.visualize import visualize_two_bbox_set

if is_interactive():
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')

# %%
config = Config.fromfile('configs/postprocess/postprocess.py')['config']
validation_results = batch_inference(config['detection_models'], None, nocache=False)

# %%
model_metadata = pd.read_csv(config['meta_file_path'])
model_metadata = model_metadata.set_index('image_id')
model_metadata.fillna(0, inplace=True)
model_metadata.loc[model_metadata['class_name'] == 'No finding', ['x_max', 'y_max']] = 1.0

gt_data, result_data = defaultdict(list), defaultdict(list)
for model, group in groupby(sorted(validation_results, key=lambda x: x[0]), key=lambda x: x[0]):
    for val_data in group:
        with open(val_data[2]) as reader:
            val_ids = [Path(line.strip()).stem for line in reader.readlines()]
        for val_id in val_ids:
            slc = model_metadata.loc[[val_id]]
            origin_img_shape = (slc.iloc[0]['original_width'], slc.iloc[0]['original_height'])
            gt_data[model].append((val_id, origin_img_shape, generate_gt_boxes(slc)))
        predicted_boxes = predict_boxes(val_ids, model_metadata, val_data[3], val_data[1])
        result_data[model] = predicted_boxes

for model in result_data.keys():
    result_data[model] = sorted(result_data[model], key=lambda x: x[0])
    gt_data[model] = sorted(gt_data[model], key=lambda x: x[0])
    assert len(gt_data[model]) == len(result_data[model])

# %%
for model in result_data.keys():
    print('Result :', model)
    measure_res = calc_measure(result_data[model], gt_data[model])

# %%
if config['grouped']:
    result_supressed = nms_models(result_data, iou_threshold=0.5)
    result_supressed_final = []
    for res in result_supressed:
        result_supressed_final.append((res[0], res[1], normal_by_boxes(*res[2], res[1])))

# %%
if config['grouped']:
    print('NMS final model: ')
    measure_res = calc_measure(result_supressed_final, gt_data[model])

# %%
if config['grouped']:
    for res1, res2, gt in zip(result_data[model], result_supressed_final, gt_data[model]):
        img_id, img_shape, bbox_data = gt[0], gt[1], gt[2]
        img = cv2.imread('data/processed/vinbigdataVOC2012/JPEGImages/' + img_id + '.png')
        img = cv2.resize(img, tuple(img_shape), interpolation=cv2.INTER_LANCZOS4)
        img_id, img_shape, bbox_data = gt[0], gt[1], gt[2]
        bboxes_gt = [np.array(rel2abs(box, img_shape)).astype(np.int) for box in bbox_data[0]]
        bboxes_res = [np.array(rel2abs(box, img_shape)).astype(np.int) for box in res1[2][0]]
        bboxes_res_nms = [np.array(rel2abs(box, img_shape)).astype(np.int) for box in res2[2][0]]
        visualize_two_bbox_set(img, (bboxes_gt, bbox_data[1], bbox_data[2]), (bboxes_res_nms, res2[2][1], res2[2][2]),
                               0.1)
        plt.show()
