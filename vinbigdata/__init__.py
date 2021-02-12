from typing import List, Tuple

BoxCoordsFloat = Tuple[float, float, float, float]
BoxWithScore = Tuple[float, float, float, float, float]
BoxCoordsInt = Tuple[int, int, int, int]
BoxesMeta = Tuple[List[BoxCoordsFloat], List[float], List[str]]
ImageMeta = Tuple[str, Tuple[int, int], BoxesMeta]

mmdetection_classes: List[Tuple[str, int]] = [('Cardiomegaly', 3), ('Aortic enlargement', 0),
                                              ('Pleural thickening', 11), ('ILD', 5), ('Nodule/Mass', 8),
                                              ('Pulmonary fibrosis', 13), ('Lung Opacity', 7), ('Atelectasis', 1),
                                              ('Other lesion', 9), ('Infiltration', 6), ('Pleural effusion', 10),
                                              ('Calcification', 2), ('Consolidation', 4), ('Pneumothorax', 12),
                                              ('No finding', 14)]

mmdetid2classname = {ind: cls[0] for ind, cls in enumerate(mmdetection_classes)}
classname2mmdetid = {cls[0]: ind for ind, cls in enumerate(mmdetection_classes)}
classname2classid = dict(mmdetection_classes)
