config = dict(
    num_gpus=2,
    detection_models=[
        dict(
            type='mmdet',
            model='results/models/detectors_equalize/latest.pth',
            config='configs/mmdetection/detectors_training.py')
    ],
    meta_file_path='data/processed/vin_dataVOC2012/train.csv',
    test_meta_file_path='data/processed/vin_dataVOC2012/test.csv',
    val_ids='data/processed/vin_dataVOC2012/image_sets/val.txt',
    test_ids='data/processed/vin_dataVOC2012/image_sets/test.txt',
    visualize=True)
