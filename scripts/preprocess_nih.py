import glob
import random
from pathlib import Path

import cv2
import pandas as pd
from mmcv import Config
from tqdm import tqdm
from vinbigdata.preprocess import ImgWriter, create_voc_dirs

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

annotations_dir, images_dir, image_sets_dir = create_voc_dirs(config['result_dir'], clear=config['clear'])

data = pd.read_csv(config['bbox_metapath'])
path_map = {Path(x).name: Path(x) for x in glob.glob(config['images_regex'])}

data['x_min'] = data['Bbox [x']
data['x_max'] = data['Bbox [x'] + data['w']
data['y_min'] = data['y']
data['y_max'] = data['y'] + data['h]']

data = data[data['Finding Label'].apply(lambda x: x in class_map.keys())]

ids = []
writer = ImgWriter(config['preprocessor'])
for path, group in tqdm(data.groupby('Image Index')):
    img_path = path_map[path]
    out_img_path = images_dir / img_path.name
    out_xml_path = annotations_dir / (img_path.stem + '.xml')

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    if img is not None:
        bboxes = [(int(bbox_meta['x_min']), int(bbox_meta['y_min']), int(bbox_meta['x_max']), int(bbox_meta['y_max']))
                  for ind, bbox_meta in group.iterrows()]
        classes = group['Finding Label'].values

        writer.process_image(img=img, bboxes=bboxes, classes=classes, image_path=out_img_path, xml_path=out_xml_path)
        ids.append(path_map[path].stem)

random.seed(221288)
random.shuffle(ids)
with open(image_sets_dir / 'train_nih.txt', 'w') as ids_writer:
    ids_writer.write('\n'.join(ids))
