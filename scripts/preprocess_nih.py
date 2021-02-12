import glob
import random
from pathlib import Path

import cv2
import pandas as pd
from mmcv import Config
from tqdm import tqdm
from vinbigdata.preprocess import ScaledYoloWriter, VocWriter

config = Config.fromfile('configs/preprocess/nih.py')['config']

class_map = {
    'Atelectasis': 'Atelectasis',
    'Effusion': 'Pleural effusion',
    'Cardiomegaly': 'Cardiomegaly',
    'Infiltrate': 'Infiltration',
    'Pneumothorax': 'Pneumothorax',
    'Mass': 'Nodule/Mass',
    'Nodule': 'Nodule/Mass'
}

data = pd.read_csv(config['bbox_metapath'])
path_map = {Path(x).name: Path(x) for x in glob.glob(config['images_regex'])}

data['x_min'] = data['Bbox [x']
data['x_max'] = data['Bbox [x'] + data['w']
data['y_min'] = data['y']
data['y_max'] = data['y'] + data['h]']

data = data[data['Finding Label'].apply(lambda x: x in class_map.keys())]

ids = []
img_writers = [
    VocWriter(config['result_dir'], config['clear'], config['preprocessor']),
    ScaledYoloWriter(config['result_dir'], False, config['preprocessor'])
]
for path, group in tqdm(data.groupby('Image Index')):
    img_path = path_map[path]

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    if img is not None:
        bboxes = [(int(bbox_meta['x_min']), int(bbox_meta['y_min']), int(bbox_meta['x_max']), int(bbox_meta['y_max']))
                  for ind, bbox_meta in group.iterrows()]
        classes = [class_map[x] for x in group['Finding Label'].values]
        for img_writer in img_writers:
            img_writer.process_image(img_name=img_path.name, img=img, bboxes=bboxes, classes=classes)
        ids.append(path_map[path].stem)

random.seed(221288)
random.shuffle(ids)
for img_writer in img_writers:
    img_writer.write_image_set(ids, 'train_nih.txt')
