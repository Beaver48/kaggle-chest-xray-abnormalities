# %%
import pandas as pd
import json
import glob
from pathlib import Path
import cv2
import random
from tqdm import tqdm
from mmcv import Config
from vinbigdata.preprocess import create_voc_dirs, ImgWriter
from vinbigdata.utils import is_interactive
from vinbigdata.visualize import plot_bboxes
import matplotlib.pyplot as plt

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
images = {Path(path).name:Path(path) for path in glob.glob(config['images'])}

# %%
annotations_dir, images_dir, image_sets_dir = create_voc_dirs(config['result_dir'], clear=config['clear'])

ids = []
writer = ImgWriter(config['preprocessor'])
for img_meta in tqdm(metadata):
    img_path = images[img_meta['file_name']]
    out_img_path = images_dir / img_path.name
    out_xml_path = annotations_dir / (img_path.stem + '.xml')
    img = cv2.imread(str(img_path))
    
    bboxes_meta = [(box, (class_map[class_name])) for box, class_name in zip(img_meta['boxes'], img_meta['syms'])
                  if class_name in class_map]
    if len(bboxes_meta) - len(img_meta) >= 2:
        continue
    if is_interactive() and config['visualize']:
        img = plot_bboxes(img, 
                          [meta[0] for meta in bboxes_meta],
                          [1.0 for meta in bboxes_meta], 
                          [meta[1] for meta in bboxes_meta])
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.show()
    writer.process_image(img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 
                         bboxes=[meta[0] for meta in bboxes_meta], 
                         classes=[meta[1] for meta in bboxes_meta], 
                         image_path=out_img_path,
                         xml_path=out_xml_path)
    ids.append(img_path.stem)

# %%
random.seed(221288)
random.shuffle(ids)
with open(image_sets_dir / 'train_chestxdet.txt', 'w') as ids_writer:
    ids_writer.write('\n'.join(ids))
