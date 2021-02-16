config = dict(
    detection_models=[
        #dict(
        #    type='mmdet',
        #    model='results/models/detectors_equalize/latest.pth',
        #    config='configs/mmdetection/detectors_training.py',
        #    num_gpu=2),
        dict(
            type='scaled_yolo',
            model='runs2/exp15/weights/best.pt',
            config='configs/yolo/data.yaml',
            img_shape=1024
        )
    ],
    meta_file_path='data/processed/vinbigdataVOC2012/train.csv',
    test_meta_file_path='data/processed/vinbigdataVOC2012/test.csv',
    val_ids='data/processed/vinbigdataVOC2012/image_sets/val.txt',
    test_ids='data/processed/vinbigdataVOC2012/image_sets/test.txt',
    visualize=True)
