from abc import ABC, abstractmethod
import numpy as np
from pascal_voc_writer import Writer
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import os
import cv2

class ImageTransform(ABC):
    def __init__(self, config: Dict[object, object]):
        self.config = config
    
    @abstractmethod
    def process(self, 
                img: np.array, 
                bboxes: List[Tuple[int,int,int,int]]) -> Tuple[np.array, List[Tuple[int,int,int,int]]]:
        raise NotImplementedError
        
class GrayscaleTransform(ImageTransform):
    def __init__(self, config: Dict[object, object]):
        super(GrayscaleTransform, self).__init__(config)
        
    def process(self, 
                img: np.array, 
                bboxes: List[Tuple[int,int,int,int]]) -> Tuple[np.array, List[Tuple[int,int,int,int]]]:
        return img, bboxes

class ImgWriter(object):
    def __init__(self, image_prepocessor: ImageTransform):
        self.image_prepocessor = image_prepocessor
       
    def process_image(self, img: np.array, 
                      bboxes: List[Tuple[int,int,int,int]],
                      classes: List[str],
                      image_path: Path,
                      xml_path: Path) -> None:
        img, bboxes = self.image_prepocessor.process(img, bboxes)
        cv2.imwrite(str(image_path), img)
        self.write_xml(xml_path, image_path, bboxes, classes, img.shape[0:2])
        


    def write_xml(self, 
                  xml_path: Path,
                  image_path: Path,
                  bboxes: List[Tuple[int,int,int,int]],
                  classes: List[str],
                  img_shape: Tuple[int, int]) -> None:
        writer = Writer(image_path, img_shape[1], img_shape[0])
        if bboxes is not None:
            for bbox, class_name in zip(bboxes, classes):
                if bbox[3] > img_shape[0] or bbox[1] > img_shape[1]:
                    continue 
                writer.addObject(class_name, 
                                 int(bbox[0]), int(bbox[1]), 
                                 int(bbox[2]), int(bbox[3]))
        writer.save(xml_path)
        