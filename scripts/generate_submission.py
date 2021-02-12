# %%
from collections import defaultdict
from itertools import groupby

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from mmcv import Config
from vinbigdata.postprocess import nms_models, normal_by_boxes
from vinbigdata.testing import batch_inference, classname2classid, predict_boxes
from vinbigdata.utils import is_interactive, rel2abs
from vinbigdata.visualize import plot_bboxes

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %%
config = Config.fromfile('configs/postprocess/postprocess.py')['config']
test_results = batch_inference(config['detection_models'], config['test_ids'], config['num_gpus'])

# %%
model_metadata = pd.read_csv(config['test_meta_file_path'])
model_metadata = model_metadata.set_index('img_id')
result_test_data = defaultdict(list)
for model, group in groupby(sorted(test_results, key=lambda x: x[0]), key=lambda x: x[0]):
    for test_data in group:
        with open(test_data[1]) as reader:
            test_ids = [line.strip() for line in reader.readlines()]
        predicted_boxes = predict_boxes(test_ids, model_metadata, test_data[2])
        result_test_data[model] = predicted_boxes

# %%
result_supressed = nms_models(result_test_data)
result_supressed_final = []
for res in result_supressed:
    result_supressed_final.append((res[0], res[1], normal_by_boxes(*res[2], res[1])))

# %%
submit = []
for img_id, img_shape, bbox_data in result_supressed_final:
    predict_str = ''
    for bbox, score, label in zip(*bbox_data):
        x_min, y_min, x_max, y_max = np.array(rel2abs(bbox, img_shape)).astype(np.int)
        class_id = classname2classid[label]
        predict_str += f' {class_id} {score} {x_min} {y_min} {x_max} {y_max}'
    submit.append({'image_id': img_id, 'PredictionString': predict_str.strip()})
    if config['visualize'] and is_interactive():
        img = cv2.imread('data/processed/vin_dataVOC2012/JPEGImages/' + img_id + '.png')
        img = cv2.resize(img, tuple(img_shape), interpolation=cv2.INTER_LANCZOS4)
        bboxes = [np.array(rel2abs(box, img_shape)).astype(np.int) for box in bbox_data[0]]
        img = plot_bboxes(img, bboxes, bbox_data[1], bbox_data[2])
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.show()
submit_df = pd.DataFrame.from_records(submit)
assert len(submit_df) == 3000
submit_df.to_csv('results/submission.csv', index=False)
