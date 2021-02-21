import copy
from typing import Dict, List, Tuple

import numpy as np
from ensemble_boxes import nms
from vinbigdata import BoxCoordsFloat, BoxesMeta, ImageMeta, classname2mmdetid, mmdetid2classname
from vinbigdata.utils import abs2rel


def nms_models(data: Dict[str, List[ImageMeta]], iou_threshold: float = 0.5) -> List[ImageMeta]:
    result_suppressed = []
    for img_index in range(len(data[list(data.keys())[0]])):
        bboxes, scores, labels, weights = [], [], [], []
        for model in data.keys():
            tupl = data[model][img_index][2]
            bboxes.append(tupl[0] if len(tupl[0]) > 0 else np.zeros((0, 4)))
            scores.append(tupl[1])
            labels.append([classname2mmdetid[label] for label in tupl[2]])
            weights.append(1.0)
        if sum([len(arr) for arr in bboxes]) != 0:
            boxes_final, scores_final, labels_final = nms(bboxes, scores, labels, iou_thr=iou_threshold)
            labels_final = [mmdetid2classname[label] for label in labels_final]
        else:
            boxes_final, scores_final, labels_final = [], [], []
        dat = data[list(data.keys())[0]][img_index]
        result_suppressed.append((dat[0], dat[1], (list(boxes_final), list(scores_final), list(labels_final))))
    return result_suppressed


def normal_by_boxes(bboxes: List[BoxCoordsFloat], scores: List[float], labels: List[str],
                    img_shape: Tuple[int, int]) -> BoxesMeta:
    bboxes, scores, labels = copy.deepcopy(bboxes), copy.deepcopy(scores), copy.deepcopy(labels)
    bboxes.append(np.array(abs2rel((0, 0, 1, 1), img_shape)))
    if len(scores) == 0:
        scores.append(1.0)
    else:
        scores.append(1 - np.max(scores))
    labels.append('No finding')
    return (bboxes, scores, labels)
