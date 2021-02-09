dataset_type = 'VOCDataset'
data_root = 'data/processed/vin_dataVOC2012/'

img_norm_cfg = dict(mean=[128, 128, 128], std=[60, 60, 60], to_rgb=True)

albumentation_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=7, p=0.5),
    dict(type='RandomBrightnessContrast', brightness_limit=0.05, contrast_limit=0.05, p=0.33),
    dict(type='RandomGamma', p=0.33)
]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albumentation_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', direction='horizontal', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('Cardiomegaly', 'Aortic enlargement', 'Pleural thickening', 'ILD', 'Nodule/Mass', 'Pulmonary fibrosis',
           'Lung Opacity', 'Atelectasis', 'Other lesion', 'Infiltration', 'Pleural effusion', 'Calcification',
           'Consolidation', 'Pneumothorax')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=[data_root + 'image_sets/train_nih.txt', data_root + 'image_sets/train_vin.txt'],
        img_prefix=data_root,
        filter_empty_gt=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'image_sets/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'image_sets/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
