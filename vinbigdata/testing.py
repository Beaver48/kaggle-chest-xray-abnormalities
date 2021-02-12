import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import checksum
import numpy as np
import pandas as pd
from IPython import get_ipython
from mmdet.core import eval_map
from tqdm import tqdm
from vinbigdata import BoxCoordsFloat, BoxesMeta, BoxWithScore, ImageMeta, classname2mmdetid, mmdetid2classname
from vinbigdata.utils import abs2rel, rel2abs


def generate_gt_boxes(img_data: pd.DataFrame) -> BoxesMeta:
    boxes, scores, labels = [], [], []
    for _, row in img_data.iterrows():
        boxes.append(
            abs2rel((row['x_min'], row['y_min'], row['x_max'], row['y_max']),
                    (row['original_width'], row['original_height'])))
        scores.append(1.0)
        labels.append(row['class_name'])
    return (boxes, scores, labels)


def batch_inference(models: List[Dict[str, str]], ids_file: str, num_gpu: int) -> List[Tuple[str, str, str]]:
    tool = '/mmdetection/tools/dist_test.sh'
    results = []

    for model_data in tqdm(models, total=len(models)):
        command = 'bash {} {} {} {} --eval=mAP --cfg-options data.test.ann_file={} \
                   --eval-options="iou_thr=0.4" --out={}'

        model_hash = checksum.get_for_file(model_data['model'])
        ids_hash = checksum.get_for_file(ids_file)
        file_name = 'results/data/result-{}-{}.pkl'.format(model_hash, ids_hash)
        if model_data['type'] == 'mmdet':
            if not Path(file_name).exists():
                command = command.format(tool, model_data['config'], model_data['model'], num_gpu, ids_file, file_name)
                res = get_ipython().run_line_magic('sx', command)
                print(res[-100:])
            results.append((model_data['model'], ids_file, file_name))
    return results


def convert_mmdet2arrays(bboxes: List[List[BoxWithScore]], img_shape: Tuple[int, int]) -> BoxesMeta:
    boxes, scores, labels = [], [], []
    for class_id, class_bboxes in enumerate(bboxes):
        for bbox in class_bboxes:
            boxes.append(abs2rel(bbox, img_shape))
            scores.append(bbox[4])
            labels.append(mmdetid2classname[class_id])
    return (boxes, scores, labels)


def convert_array2mmdet(boxes: List[BoxCoordsFloat],
                        scores: List[float],
                        labels: List[str],
                        num_classes=15) -> List[List[BoxWithScore]]:
    result: List[List[BoxWithScore]] = [[] for _ in range(num_classes)]
    for box, score, label in zip(boxes, scores, labels):
        result[classname2mmdetid[label]].append(np.array([*box, score]))
    result = [np.array(res) if len(res) > 0 else np.zeros((0, 5)) for res in result]

    return result


def predict_boxes(img_ids: List[str], meta: pd.DataFrame, file_path: str) -> List[ImageMeta]:
    predict_bboxes = pickle.load(open(file_path, 'rb'))

    img_shapes: List[Tuple[int, int]] = [meta[['width', 'height']].loc[[img_id]].values[0] for img_id in img_ids]
    original_img_shapes: List[Tuple[int, int]] = [
        meta[['original_width', 'original_height']].loc[[img_id]].values[0] for img_id in img_ids
    ]
    res = []
    for img_id, img_shape, original_img_shape, predict_boxes_img in zip(img_ids, img_shapes, original_img_shapes,
                                                                        predict_bboxes):
        res.append((img_id, original_img_shape, convert_mmdet2arrays(predict_boxes_img, img_shape)))
    return res


def calc_measure(result: List[ImageMeta], gt_result: List[ImageMeta]) -> Tuple[float, Dict[str, float]]:
    x = [convert_array2mmdet([rel2abs(box, res[1]) for box in res[2][0]], res[2][1], res[2][2]) for res in result]
    y = [{
        'bboxes': np.array([rel2abs(box, res[1]) for box in res[2][0]]) if len(res[2][0]) > 0 else np.zeros((0, 4)),
        'labels': np.array([classname2mmdetid[x] for x in res[2][2]])
    } for res in gt_result]
    res = eval_map(x, y, dataset=list(mmdetid2classname.values()), iou_thr=0.4)
    return res
