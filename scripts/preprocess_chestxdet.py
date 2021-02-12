# %%
import glob
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mmcv import Config
from tqdm import tqdm
from vinbigdata.preprocess import ScaledYoloWriter, VocWriter
from vinbigdata.utils import is_interactive
from vinbigdata.visualize import plot_bboxes

# %%
class_map = {
    'Atelectasis': 'Atelectasis',
    'Effusion': 'Pleural effusion',
    'Calcification': 'Calcification',
    'Consolidation': 'Consolidation',
    'Fibrosis': 'Pulmonary fibrosis',
    'Pneumothorax': 'Pneumothorax',
    'Mass': 'Nodule/Mass',
    'Nodule': 'Nodule/Mass'
}

# %%
config = Config.fromfile('configs/preprocess/chestxdet.py')['config']
metadata = []
for f in glob.glob(config['metapath']):
    metadata.extend(json.load(open(f, 'r')))
images = {Path(path).name: Path(path) for path in glob.glob(config['images'])}

# %%
img_writers = [
    VocWriter(config['result_dir'], config['clear'], config['preprocessor']),
    ScaledYoloWriter(config['result_dir'], False, config['preprocessor'])
]
ids = []
for img_meta in tqdm(metadata):
    img_path = images[img_meta['file_name']]
    img = cv2.imread(str(img_path))

    bboxes_meta = [(box, (class_map[class_name])) for box, class_name in zip(img_meta['boxes'], img_meta['syms'])
                   if class_name in class_map]
    if len(bboxes_meta) - len(img_meta) >= 2:
        continue
    if is_interactive() and config['visualize']:
        img = plot_bboxes(img, [meta[0] for meta in bboxes_meta], [1.0 for meta in bboxes_meta],
                          [meta[1] for meta in bboxes_meta])
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()
    for writer in img_writers:
        writer.process_image(
            img_name=img_path.name,
            img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            bboxes=[meta[0] for meta in bboxes_meta],
            classes=[meta[1] for meta in bboxes_meta])
    ids.append(img_path.stem)

# %%
random.seed(221288)
random.shuffle(ids)
for writer in img_writers:
    writer.write_image_set(ids, 'train_chestxdet.txt')
