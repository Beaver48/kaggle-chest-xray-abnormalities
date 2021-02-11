config = dict(
    num_gpus=4,
    detection_models=[dict(type='mmdet', 
                           model='results/models/detectors_clear_3channel/latest.pth',
                           config='configs/mmdetection/detectors_training.py'),
                      dict(type='mmdet', 
                           model='results/models/detectors_clear_3channel/epoch_45.pth',
                           config='configs/mmdetection/detectors_training.py')
                     ],
    meta_file_path='data/processed/vin_dataVOC2012/train.csv',
    val_ids='data/processed/vin_dataVOC2012/image_sets/val.txt',
    test_ids='data/processed/vin_dataVOC2012/image_sets/test.txt'
)
    
