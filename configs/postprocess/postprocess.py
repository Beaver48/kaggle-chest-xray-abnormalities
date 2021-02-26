config = dict(
    detection_models=[
        # dict(
        #    type='mmdet',
        #    model='results/models/detectors_equalize/latest.pth',
        #    config='configs/mmdetection/detectors_training.py',
        #    num_gpu=2),
        dict(
            type='scaled_yolo',
            model='results/models/scaled_yolo_normal-1000_epoch-160_only-vin.pt',
            config='configs/yolo/data.yaml',
            img_shape=1024),  # 0.457 local dirty cv, 0.285 leaderboard
        dict(
            type='scaled_yolo',
            model='results/models/scaled_yolo_normal-1000_epoch-160_vin_nih.pt',
            config='configs/yolo/data.yaml',
            img_shape=1024),  # 0.466 local dirty cv, 0.253 leaderboard
    ],
    meta_file_path='data/processed/vinbigdataVOC2012/train.csv',
    test_meta_file_path='data/processed/vinbigdataVOC2012/test.csv',
    val_ids='data/processed/vinbigdataVOC2012/image_sets/val.txt',
    test_ids='data/processed/vinbigdataVOC2012/image_sets/test.txt',
    visualize=False)
