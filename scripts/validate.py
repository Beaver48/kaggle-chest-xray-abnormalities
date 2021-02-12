# %%
from collections import defaultdict
from itertools import groupby

import pandas as pd
from IPython import get_ipython
from mmcv import Config
from vinbigdata.postprocess import nms_models, normal_by_boxes
from vinbigdata.testing import batch_inference, calc_measure, generate_gt_boxes, predict_boxes
from vinbigdata.utils import is_interactive

if is_interactive():
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')

# %%
config = Config.fromfile('configs/postprocess/postprocess.py')['config']
validation_results = batch_inference(config['detection_models'], config['val_ids'], config['num_gpus'])

# %%
model_metadata = pd.read_csv(config['meta_file_path'])
model_metadata = model_metadata.set_index('image_id')
model_metadata.fillna(0, inplace=True)
model_metadata.loc[model_metadata['class_name'] == 'No finding', ['x_max', 'y_max']] = 1.0

gt_data, result_data = defaultdict(list), defaultdict(list)
for model, group in groupby(sorted(validation_results, key=lambda x: x[0]), key=lambda x: x[0]):
    for val_data in group:
        with open(val_data[1]) as reader:
            val_ids = [line.strip() for line in reader.readlines()]
        for val_id in val_ids:
            slc = model_metadata.loc[[val_id]]
            origin_img_shape = (slc.iloc[0]['original_width'], slc.iloc[0]['original_height'])
            gt_data[model].append((val_id, origin_img_shape, generate_gt_boxes(slc)))
        predicted_boxes = predict_boxes(val_ids, model_metadata, val_data[2])
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
result_supressed = nms_models(result_data)
result_supressed_final = []
for res in result_supressed:
    result_supressed_final.append((res[0], res[1], normal_by_boxes(*res[2], res[1])))

# %%
print('NMS final model: ')
measure_res = calc_measure(result_supressed_final, gt_data[model])
