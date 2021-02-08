import numba
import shutil
from pathlib import Path
from typing import Tuple

@numba.jit
def compute_intersection_area(bbox1:Tuple[float, float, float, float], 
                              bbox2:Tuple[float, float, float, float]) -> float:
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
def compute_union_area(bbox1:Tuple[float, float, float, float], 
                       bbox2:Tuple[float, float, float, float]) -> float:
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


def create_voc_dirs(data_dir: str, clear: bool=False) -> Tuple[Path, Path, Path]:
    base_dir = Path(data_dir)
    if clear and base_dir.exists():
        shutil.rmtree(base_dir)
    annotations = base_dir / "Annotations"
    images = base_dir / "JPEGImages"
    image_sets =  base_dir / "image_sets"
    
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    image_sets.mkdir(parents=True, exist_ok=True)
    return (annotations, images, image_sets)