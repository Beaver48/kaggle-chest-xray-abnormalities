from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pascal_voc_writer import Writer
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut


class ImageTransform(ABC):
    """ Basic transformation
    """

    def __init__(self, config: Dict[object, object]):
        self.config = config

    @abstractmethod
    def process(self, img: np.array, bboxes: List[Tuple[int, int, int,
                                                        int]]) -> Tuple[np.array, List[Tuple[int, int, int, int]]]:
        raise NotImplementedError


class GrayscaleTransform(ImageTransform):
    """ Transformation for grayscale
    """

    def __init__(self, config: Dict[object, object]):
        super(GrayscaleTransform, self).__init__(config)

    def process(self, img: np.array, bboxes: List[Tuple[int, int, int,
                                                        int]]) -> Tuple[np.array, List[Tuple[int, int, int, int]]]:
        return img, bboxes


class ImgWriter(object):
    """ Class for writing preprocessed data in PASCALVOC 2012 format
    """

    def __init__(self, image_prepocessor: ImageTransform):
        self.image_prepocessor = image_prepocessor

    def process_image(self, img: np.array, bboxes: List[Tuple[int, int, int, int]], classes: List[str],
                      image_path: Path, xml_path: Path) -> None:
        img, bboxes = self.image_prepocessor.process(img, bboxes)
        cv2.imwrite(str(image_path), img)
        self.write_xml(xml_path, image_path, bboxes, classes, img.shape[0:2])

    def write_xml(self, xml_path: Path, image_path: Path, bboxes: List[Tuple[int, int, int, int]], classes: List[str],
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
