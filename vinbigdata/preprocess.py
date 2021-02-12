import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import glob
from pathlib import Path
from albumentations import BboxParams, Compose, Resize
from pascal_voc_writer import Writer
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing_extensions import TypedDict
from vinbigdata import BoxCoordsInt

ImageMeta = TypedDict('ImageMeta', {
    'image_id': str,
    'class_name': str,
    'rad_id': Optional[str],
    'x_min': int,
    'y_min': int,
    'x_max': int,
    'y_max': int
})


class BaseTransform(ABC):
    """ Base transformation
    """

    def __init__(self, fized_size: Tuple[int, int]) -> None:
        self.resize_transform = Compose([Resize(fized_size[0], fized_size[0], always_apply=True)],
                                        bbox_params=BboxParams(
                                            format='pascal_voc', min_visibility=0.0, label_fields=['classes']))

    @abstractmethod
    def __call__(self, img_name:str, img: np.array, bboxes: List[BoxCoordsInt],
                 classes: List[str]) -> Tuple[np.array, List[BoxCoordsInt], List[str]]:
        raise NotImplementedError('call method not implemented')


class GrayscaleTransform(BaseTransform):
    """ Transformation for grayscale
    """
    def __init__(self, fized_size: Tuple[int, int] = (1024, 1024))-> None:
        super(GrayscaleTransform, self).__init__(fized_size)
    def __call__(self, img_name:str, 
                 img: np.array, bboxes: List[BoxCoordsInt],
                 classes: List[str]) -> Tuple[np.array, List[BoxCoordsInt], List[str]]:
        assert len(img.shape) == 2
        res = self.resize_transform(image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
                                    bboxes=bboxes, 
                                    classes=classes)
        return (res['image'], res['bboxes'], res['classes'])

class MaskTransform(BaseTransform):
    """ Transformation for grayscale
    """
    def __init__(self, mask_path: str,
                 fized_size:Tuple[int, int] = (1024, 1024)) -> None:
        super(MaskTransform, self).__init__(fized_size)
        self.masks = {Path(mask).name: mask for mask in glob.glob(mask_path + "/*")}
        
    def __call__(self, 
                 img_name:str, 
                 img: np.array, 
                 bboxes: List[BoxCoordsInt],
                 classes: List[str]) -> Tuple[np.array, List[BoxCoordsInt], List[str]]:
        mask = cv2.resize(cv2.imread(self.masks[Path(img_name).name]),
                          (img.shape[1], img.shape[0]), 
                           interpolation=cv2.INTER_LANCZOS4)[:,:,0]
        assert len(mask.shape) == 2
        img = np.concatenate(
            [img[:, :, np.newaxis],
             img[:, :, np.newaxis],
             mask[:, :, np.newaxis]], axis=2)
        res = self.resize_transform(image=img, bboxes=bboxes, classes=classes)
        assert len(img.shape) == 3
        return (res['image'], res['bboxes'], res['classes'])

class EqualizeTransform(BaseTransform):
    """ Transformation with equalization
    """

    def __init__(
        self,
        clahe_clip_limit: float = 4.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        fized_size: Tuple[int, int] = (1024, 1024)
    ) -> None:
        super(EqualizeTransform, self).__init__(fized_size)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid = clahe_grid

    def __call__(self, img_name:str, img: np.array, bboxes: List[BoxCoordsInt],
                 classes: List[str]) -> Tuple[np.array, List[BoxCoordsInt], List[str]]:
        assert len(img.shape) == 2
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid)
        img = np.concatenate(
            [img[:, :, np.newaxis],
             cv2.equalizeHist(img)[:, :, np.newaxis],
             clahe.apply(img)[:, :, np.newaxis]],
            axis=2)
        res = self.resize_transform(image=img, bboxes=bboxes, classes=classes)
        return res['image'], res['bboxes'], res['classes']


class ImgWriter:
    """ Class for writing data in PASCALVOC 2012 format
    """

    def __init__(self, image_prepocessor: BaseTransform) -> None:
        self.image_prepocessor = image_prepocessor

    def process_image(self, img: np.array, bboxes: List[BoxCoordsInt], classes: List[str], image_path: Path,
                      xml_path: Path) -> Tuple[int, int]:
        img, bboxes, classes = self.image_prepocessor(image_path.name, img, bboxes, classes)
        cv2.imwrite(str(image_path), img)
        self.write_xml(xml_path, image_path, bboxes, classes, img.shape[0:2])
        return img.shape

    def write_xml(self, xml_path: Path, image_path: Path, bboxes: List[BoxCoordsInt], classes: List[str],
                  img_shape: Tuple[int, int]) -> None:
        writer = Writer(image_path, img_shape[1], img_shape[0])
        if bboxes is not None:
            for bbox, class_name in zip(bboxes, classes):
                if bbox[3] > img_shape[0] or bbox[1] > img_shape[1]:
                    continue
                writer.addObject(class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        writer.save(xml_path)


def read_dicom_img(path: str, apply_voi: bool = True) -> np.array:
    dicom_data = dcmread(path)
    if apply_voi:
        img_data = apply_voi_lut(dicom_data.pixel_array, dicom_data)
    else:
        img_data = dicom_data.pixel_array

    if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
        img_data = np.amax(img_data) - img_data
    img_data = img_data - np.min(img_data)
    img_data = img_data / np.max(img_data)
    img_data = (img_data * 256).astype(np.uint8)
    return img_data


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


def convert_bboxmeta2arrays(bbox_metas: List[ImageMeta]) -> Tuple[List[BoxCoordsInt], List[float], List[str]]:
    bboxes = [(bbox_meta['x_min'], bbox_meta['y_min'], bbox_meta['x_max'], bbox_meta['y_max'])
              for bbox_meta in bbox_metas if bbox_meta['class_name'] != 'No finding']
    labels = [bbox_meta['class_name'] for bbox_meta in bbox_metas if bbox_meta['class_name'] != 'No finding']
    scores = [1.0 for _ in range(len(bboxes))]
    return (bboxes, scores, labels)
