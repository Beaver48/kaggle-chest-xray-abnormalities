import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numba
import numpy as np
import seaborn as sns


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


def create_voc_dirs(data_dir: str, clear: bool = False) -> Tuple[Path, Path, Path]:
    base_dir = Path(data_dir)
    if clear and base_dir.exists():
        shutil.rmtree(base_dir)
    annotations = base_dir / 'Annotations'
    images = base_dir / 'JPEGImages'
    image_sets = base_dir / 'image_sets'

    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    image_sets.mkdir(parents=True, exist_ok=True)
    return (annotations, images, image_sets)


COLOR_MAP: Dict[str, Tuple[int, int, int]] = {}
PALLET_INDEX = 0


def add_bboxes(img: np.array, bboxes: List[Tuple[int, int, int, int]], scores: List[float],
               labels: List[str]) -> np.array:
    for bbox, score, label in zip(bboxes, scores, labels):
        if label not in COLOR_MAP:
            color = sns.color_palette('tab20')[len(COLOR_MAP)]
            COLOR_MAP[label] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLOR_MAP[label], 7)
        img = cv2.putText(img, label + '%.2f' % score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                          1 * img.shape[0] / 1000, COLOR_MAP[label], int(2 * img.shape[0] / 1000))
    return img
