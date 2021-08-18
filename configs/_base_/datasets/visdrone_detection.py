# dataset settings
dataset_type = 'VisDroneDataset'
data_root = 'data/visdrone/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=[(1333, 800),(1333*1.1,800*1.1),(1333/1.2,800/1.2)], multiscale_mode='value', keep_ratio=True),# bbox_clip_border=False),
    dict(type='Resize', img_scale=(1600, 1050), keep_ratio=True),
    # dict(type=('Mosaic_OpenCV')),
    # dict(type='HSV_OpenCV'),
    # dict(type='Resize', img_scale=[(1333*1.1,800*1.1),(1333/1.2,800/1.2)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1300, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, bbox_clip_border=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # if_mosaic=True,
        ann_file=data_root + 'annotations/coco-cut_train_val_NOother_new.json',
        img_prefix=data_root + 'images/VisDrone2019-DET-train_val/images-cut-NOother-new',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco-test-dev.json',
        img_prefix=data_root + 'images/VisDrone2019-DET-test-dev/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco-test-dev.json',
        img_prefix=data_root + 'images/VisDrone2019-DET-test-dev/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
