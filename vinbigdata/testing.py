import pandas as pd
from typing import List, Tuple, Dict
from vinbigdata.utils import abs2rel
from tqdm import tqdm
import checksum
from pathlib import Path 

mmdetection_classes: List[Tuple[str, int]] = [
    ('Cardiomegaly', 3),
    ('Aortic enlargement', 0),
    ('Pleural thickening', 11),
    ('ILD', 5),
    ('Nodule/Mass', 8),
    ('Pulmonary fibrosis', 13),
    ('Lung Opacity', 7),
    ('Atelectasis', 1),
    ('Other lesion', 9),
    ('Infiltration', 6),
    ('Pleural effusion', 10),
    ('Calcification', 2),
    ('Consolidation', 4),
    ('Pneumothorax', 12),
    ('No finding', 14)]
    
mmdetid2classname = dict([(ind, cls[0]) for ind, cls in enumerate(mmdetection_classes)])
classname2mmdetid = dict([(cls[0], ind) for ind, cls in enumerate(mmdetection_classes)])
classname2classid = dict(mmdetection_classes)

def generate_gt_boxes(img_data: pd.DataFrame) -> Tuple[List[Tuple[float, float, float, float]], 
                                                   List[float],
                                                   List[str]
                                              ]:
    boxes, scores, labels = [], [], []
    for ind, row in img_data.iterrows():
        boxes.append(abs2rel((row['x_min'], row['y_min'], row['x_max'], row['y_max']), 
                             (row['original_width'], row['original_height'])))
        scores.append(1.0)
        labels.append(row['class_name'])
    return (boxes, scores, labels)
                     
def batch_inference(models: List[Dict[str, str]], ids_file: str, num_gpu: int) -> List[Tuple[str, str, str]]:
    tool = '/mmdetection/tools/dist_test.sh'
    command = 'bash {} {} {} {} --eval=mAP --cfg-options data.test.ann_file={} --eval-options="iou_thr=0.4" --out={}'

    results = []
    for model_ind, model_data in tqdm(enumerate(models), total=len(models)):
        model_hash = checksum.get_for_file(model_data['model'])
        ids_hash = checksum.get_for_file(ids_file)
        if model_data['type'] == 'mmdet':
            file_name = 'results/data/result-{}-{}.pkl'.format(model_hash, ids_hash)
            if not Path(file_name).exists():
                command = command.format(tool, 
                                         model_data['config'],
                                         model_data['model'],
                                         num_gpu,
                                         ids_file, 
                                         file_name)
                res = get_ipython().run_line_magic("sx", command)
                print(res)
            results.append((model_data['model'], ids_file, file_name))
    return results