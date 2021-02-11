from typing import Tuple, List, Dict

import cv2
import numba
import numpy as np



def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


@numba.jit
def compute_intersection_area(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> int:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    h = x2 - x1
    w = y2 - y1
    if h < 0:
        h = 0
    if w < 0:
        h = 0
    return h * w


@numba.jit
def compute_area(bbox: Tuple[int, int, int, int]) -> int:
    return (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])


@numba.jit
def compute_union_area(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> int:
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    h = x2 - x1
    w = y2 - y1
    if h < 0:
        h = 0
    if w < 0:
        h = 0
    return h * w


def resize(img: np.array, max_size: Tuple[int, int]) -> np.array:
    return cv2.resize(img, max_size, interpolation=cv2.INTER_LANCZOS4)


@numba.jit
def rel2abs(box: Tuple[float, float, float, float],
            img_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    return (box[0] * img_shape[0], 
              box[1] * img_shape[1], 
               box[2] * img_shape[0], 
               box[3] * img_shape[1])

@numba.jit
def abs2rel(box: Tuple[float, float, float, float],
                     img_shape: Tuple[int, int]
                    ) -> Tuple[float, float, float, float]:
    return(box[0] / img_shape[0], 
              box[1] / img_shape[1], 
               box[2] / img_shape[0], 
               box[3] / img_shape[1])


def convert_mmdet2arrays(bboxes:List[List[Tuple[float, float, float, float, float]]], 
                         img_shape: Tuple[int, int])  -> Tuple[List[Tuple[float, float, float, float]], List[float], List[str]]:
    boxes, scores, labels = [], [], []
    for class_id, class_bboxes in enumerate(bboxes):
        for bbox in class_bboxes:
            boxes.append(abs2rel(bbox, img_shape))
            scores.append(bbox[4])
            labels.append(mmdet2class[class_id])
    return (boxes, scores, labels)

def convert_array2mmdet(boxes:List[Tuple[float, float, float, float]], 
                        scores:List[float], 
                        labels:List[str], 
                        num_classes=15) -> List[List[Tuple[float, float, float, float, float]]]:
    result = [[] for _ in range(num_classes)]
    for box, score, label in zip(boxes, scores, labels):
        result[class2mmdet[label]].append(np.array([*box, score]))
    result = [np.array(res) if len(res) > 0 else np.zeros((0,5)) for res in result]
    
    return result