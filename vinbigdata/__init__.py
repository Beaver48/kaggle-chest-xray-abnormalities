from typing import List, Tuple

BoxCoordsFloat = Tuple[float, float, float, float]
BoxWithScore = Tuple[float, float, float, float, float]
BoxCoordsInt = Tuple[int, int, int, int]
BoxesMeta = Tuple[List[BoxCoordsFloat], List[float], List[str]]
ImageMeta = Tuple[str, Tuple[int, int], BoxesMeta]
