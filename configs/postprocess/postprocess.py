config = dict(
    num_gpus=4,
    detection_models=[dict(type='mmdet', 
                           model='results/models/detectors_clear_3channel/epoch_3.pth',
                           config='configs/mmdetection/detectors_training.py'),
                      dict(type='mmdet', 
                           model='results/models/detectors_clear_3channel/epoch_2.pth',
                           config='configs/mmdetection/detectors_training.py')
                     ],
    meta_file_path='data/processed/vin_dataVOC2012/train.csv',
    val_ids='data/processed/vin_dataVOC2012/image_sets/val.txt',
)
    
