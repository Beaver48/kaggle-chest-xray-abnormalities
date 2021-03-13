# %%
import glob
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from mmcv import Config
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from typing_extensions import TypedDict
from vinbigdata.preprocess import BaseWriter, ImageMeta, ScaledYoloWriter, VocWriter, read_dicom_img
from vinbigdata.utils import compute_area, compute_intersection_area, compute_union_area, is_interactive
from vinbigdata.visualize import visualize_label_suppression

# %%
config = Config.fromfile('configs/preprocess/vin.py')['config']
train_data = pd.read_csv(config['train_metapath'])
train_data['file_name'] = train_data['image_id'].apply(lambda x: Path(config['train_images_dir']) / (x + '.dicom'))

images = []
for k, g in groupby(train_data.sort_values('file_name').to_dict(orient='records'), lambda x: x['file_name']):
    images.append((k, sorted(list(g), key=lambda x: (x['class_id'], x['rad_id']))))


# %%
def consolidate_radiologists(img_meta: List[ImageMeta],
                             rad_confidence: Dict[str, float],
                             img_shape: Tuple[int, int],
                             iou_threshold: float = 0.10) -> List[ImageMeta]:
    new_meta = []
    image_id = img_meta[0]['image_id']
    for class_name, group in groupby(sorted(img_meta, key=lambda x: x['class_name']), lambda x: x['class_name']):
        bboxes_mt = list(group)
        if class_name == 'No finding':
            if len(bboxes_mt) == len(img_meta):
                bbox = ImageMeta({
                    'image_id': image_id,
                    'class_name': class_name,
                    'rad_id': None,
                    'x_min': bboxes_mt[0]['x_min'],
                    'y_min': bboxes_mt[0]['y_min'],
                    'x_max': bboxes_mt[0]['x_max'],
                    'y_max': bboxes_mt[0]['y_max']
                })
                new_meta.append(bbox)
            continue
        bboxes = [[[
            bboxes_mt[i]['x_min'] / img_shape[1], bboxes_mt[i]['y_min'] / img_shape[0],
            bboxes_mt[i]['x_max'] / img_shape[1], bboxes_mt[i]['y_max'] / img_shape[0]
        ] for i in range(len(bboxes_mt))]]
        scores = [[rad_confidence[bboxes_mt[i]['rad_id']] * 2 for i in range(len(bboxes_mt))]]
        labels = [[1 for i in range(len(bboxes_mt))]]
        boxes_final, _, _ = weighted_boxes_fusion(bboxes, scores, labels, iou_thr=iou_threshold)

        for processed_bbox in boxes_final:
            bbox = ImageMeta({
                'image_id': image_id,
                'class_name': class_name,
                'rad_id': None,
                'x_min': processed_bbox[0] * img_shape[1],
                'y_min': processed_bbox[1] * img_shape[0],
                'x_max': processed_bbox[2] * img_shape[1],
                'y_max': processed_bbox[3] * img_shape[0]
            })
            new_meta.append(bbox)
    return new_meta


def calc_rad_confidence(image_metas: List[Tuple[Path, List[ImageMeta]]],
                        iou_threshold: float = 0.5) -> Dict[str, float]:
    radiologist_cnt: Dict[str, int] = defaultdict(int)
    radiologist_up: Dict[str, int] = defaultdict(int)
    radiologists_confidence: Dict[str, float] = defaultdict(float)
    for _, bboxes_meta in image_metas:
        for x in bboxes_meta:
            if x['class_name'] != 'No finding':
                radiologist_cnt[x['rad_id']] += 1
    for _, bboxes_meta in image_metas:
        for class_name, group in groupby(sorted(bboxes_meta, key=lambda x: x['class_name']), lambda x: x['class_name']):
            if class_name == 'No finding':
                continue
            bboxes_mt = list(group)
            bboxes = [(bboxes_mt[i]['x_min'], bboxes_mt[i]['y_min'], bboxes_mt[i]['x_max'], bboxes_mt[i]['y_max'])
                      for i in range(len(bboxes_mt))]
            bbox_set = set()
            for bbox1_ind, bbox1 in enumerate(bboxes):
                for bbox2_ind, bbox2 in enumerate(bboxes):
                    sim = compute_intersection_area(bbox1, bbox2) / compute_union_area(bbox1, bbox2)
                    if sim > iou_threshold and bboxes_mt[bbox1_ind]['rad_id'] != bboxes_mt[bbox2_ind]['rad_id']:
                        radiologist_up[bboxes_mt[bbox1_ind]['rad_id']] += 1
                        bbox_set.add(bbox1_ind)
                        bbox_set.add(bbox2_ind)
                        break
    for rad_id in radiologist_cnt.keys():
        radiologists_confidence[rad_id] = radiologist_up[rad_id] / radiologist_cnt[rad_id]
    return radiologists_confidence


def filter_boxes_without_intersection(image_meta: Tuple[Path, List[ImageMeta]],
                                      iou_threshold: float = 0.3,
                                      io_sim_threshols: float = 0.9) -> Tuple[Path, List[ImageMeta]]:
    img_path = image_meta[0]
    filtered_image_meta = []
    for class_name, group in groupby(sorted(image_meta[1], key=lambda x: x['class_name']), lambda x: x['class_name']):
        bboxes_mt = list(group)
        if class_name == 'No finding':
            filtered_image_meta.append(bboxes_mt[0])
            continue

        bboxes = [(bboxes_mt[i]['x_min'], bboxes_mt[i]['y_min'], bboxes_mt[i]['x_max'], bboxes_mt[i]['y_max'])
                  for i in range(len(bboxes_mt))]
        bbox_set = set()
        for bbox1_ind, bbox1 in enumerate(bboxes):
            for bbox2_ind, bbox2 in enumerate(bboxes):
                if bboxes_mt[bbox1_ind]['rad_id'] == bboxes_mt[bbox2_ind]['rad_id']:
                    continue
                sim = compute_intersection_area(bbox1, bbox2) / compute_union_area(bbox1, bbox2)
                area1 = compute_area(bbox1)
                area2 = compute_area(bbox2)
                sim_inside = compute_intersection_area(bbox1, bbox2) / min(area1, area2)
                if ((sim > iou_threshold) and (bbox1_ind not in bbox_set)) or ((sim_inside > io_sim_threshols) and
                                                                               (area1 < area2)):
                    bbox_set.add(bbox1_ind)
                    filtered_image_meta.append(bboxes_mt[bbox1_ind])
                if ((sim > iou_threshold) and (bbox2_ind not in bbox_set)) or ((sim_inside > io_sim_threshols) and
                                                                               (area1 > area2)):
                    bbox_set.add(bbox2_ind)
                    filtered_image_meta.append(bboxes_mt[bbox2_ind])
    return (img_path, filtered_image_meta)


# %%
radiologists_confidence_map = calc_rad_confidence(images)


# %%
def create_pipeline(cfg: Dict[object, object]) -> Callable:

    def filter_meta(meta: Tuple[Path, List[ImageMeta]], img_shape: Tuple[int, int],
                    radiologists_confidence: Dict[str, float]) -> List[ImageMeta]:
        if cfg['aggree_rad']:
            filtered_meta = filter_boxes_without_intersection(meta)[1]
        else:
            filtered_meta = meta[1]
        if len(filtered_meta) == 0:
            return None
        filtered_meta = consolidate_radiologists(filtered_meta, radiologists_confidence, img_shape)
        return filtered_meta

    return filter_meta


pipeline_func = create_pipeline(config)

# %%
if is_interactive() and config['visualize']:
    for meta in images[0:150]:
        met = [m for m in meta[1] if m['class_name'] != 'No finding']
        if len(met) > 0:
            img = cv2.cvtColor(read_dicom_img(str(meta[0])), cv2.COLOR_GRAY2RGB)
            filtered_meta = pipeline_func(meta, img.shape, radiologists_confidence_map)
            if filtered_meta is not None:
                visualize_label_suppression(img, meta[1], filtered_meta)
                plt.show()

# %%
TrainMeta = TypedDict(
    'TrainMeta', {
        'image_id': str,
        'class_name': str,
        'rad_id': str,
        'x_min': float,
        'x_max': float,
        'y_min': float,
        'y_max': float,
        'width': int,
        'height': int,
        'original_width': int,
        'original_height': int
    })

TestMeta = TypedDict('TestMeta', {
    'img_id': str,
    'width': int,
    'height': int,
    'original_width': int,
    'original_height': int
})


def process_train(img_meta: Tuple[Path, ImageMeta], radiologists_confidence: Dict[str, float],
                  writers: List[BaseWriter]) -> TrainMeta:
    origin_image_filename = img_meta[0]
    img = read_dicom_img(str(origin_image_filename))
    img_shape = img.shape

    final_meta = pipeline_func(img_meta, img_shape, radiologists_confidence)
    if final_meta is None:
        return None
    bboxes = [(meta['x_min'], meta['y_min'], meta['x_max'], meta['y_max']) for meta in final_meta
              if meta['class_name'] != 'No finding']
    classes = [meta['class_name'] for meta in final_meta if meta['class_name'] != 'No finding']
    for writer in writers:
        new_shape = writer.process_image(
            img_name=origin_image_filename.stem + '.png', img=img, bboxes=bboxes, classes=classes)
    for meta in final_meta:
        meta['original_width'], meta['original_height'] = img_shape[1], img_shape[0]
        meta['width'], meta['height'] = new_shape[1], new_shape[0]
    return final_meta


def process_test(origin_image_filename: Path, writers: List[BaseWriter]) -> TestMeta:
    img = read_dicom_img(str(origin_image_filename))
    img_shape = img.shape
    for writer in writers:
        new_shape = writer.process_image(img_name=origin_image_filename.stem + '.png', img=img, bboxes=[], classes=[])

    return {
        'img_id': origin_image_filename.stem,
        'original_width': img_shape[1],
        'original_height': img_shape[0],
        'width': new_shape[1],
        'height': new_shape[0]
    }


# %%
img_writers = [
    VocWriter(config['result_dir'], config['clear'], config['preprocessor']),
    ScaledYoloWriter(config['result_dir'], False, config['preprocessor'])
]
train = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(lambda x: process_train(x, radiologists_confidence_map, img_writers))(dat) for dat in tqdm(images))
test = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(lambda img_path: process_test(Path(img_path), img_writers))(img_path)
    for img_path in tqdm(glob.glob(str(Path(config['test_images_dir']) / '*'))))
train = pd.DataFrame.from_records([item for sublist in train if sublist is not None for item in sublist])
train.to_csv(Path(config['result_dir']) / 'train.csv')
test = pd.DataFrame.from_records(test)
test.to_csv(Path(config['result_dir']) / 'test.csv')

# %%
from sklearn.model_selection import GroupKFold
gss = GroupKFold(n_splits=5)
for ind, (train_indecies, test_indecies) in enumerate(gss.split(train, train['class_name'], train['image_id'])):
    for writer in img_writers:
        writer.write_image_set(
            train['image_id'][train_indecies].drop_duplicates().sample(frac=1, random_state=211288).values, 'train_vin_fold_{}.txt'.format(ind))
        writer.write_image_set(
            train['image_id'][test_indecies].drop_duplicates().sample(frac=1, random_state=211288).values, 'val_fold_{}.txt'.format(ind))
for writer in img_writers:
    writer.write_image_set(train['image_id'].drop_duplicates().sample(frac=1, random_state=211288).values,
                               'all_vin.txt')
    writer.write_image_set(test['img_id'].apply(lambda x: x).values, 'test.txt')

# %%
