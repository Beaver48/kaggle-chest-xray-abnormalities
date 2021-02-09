import copy
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from vinbigdata.preprocess import ImageMeta, convert_bboxmeta2arrays

COLOR_MAP: Dict[str, Tuple[int, int, int]] = {}


def plot_bboxes(img: np.array, bboxes: List[Tuple[int, int, int, int]], scores: List[float],
                labels: List[str]) -> np.array:
    for bbox, score, label in zip(bboxes, scores, labels):
        if label not in COLOR_MAP:
            color = sns.color_palette('tab20')[len(COLOR_MAP)]
            COLOR_MAP[label] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLOR_MAP[label], 7)
        img = cv2.putText(img, label + '%.2f' % score, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX,
                          1 * img.shape[0] / 1000, COLOR_MAP[label], int(2 * img.shape[0] / 1000))
    return img


def visualize_label_suppression(img: np.array,
                                bbox_set1: List[ImageMeta],
                                bbox_set2: List[ImageMeta],
                                fig_size: Tuple[float, float] = (28, 28)) -> None:
    bboxes, scores, labels = convert_bboxmeta2arrays(bbox_set1)
    img1 = plot_bboxes(copy.deepcopy(img), bboxes, scores, labels)

    bboxes, scores, labels = convert_bboxmeta2arrays(bbox_set2)
    img2 = plot_bboxes(copy.deepcopy(img), bboxes, scores, labels)
    fig = plt.figure(figsize=fig_size)
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    return fig
