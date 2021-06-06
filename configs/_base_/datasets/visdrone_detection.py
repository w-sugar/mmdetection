# dataset settings
dataset_type = 'VisDroneDataset'
data_root = 'data/visdrone/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1333, 800),(1333*1.1,800*1.1),(1333/1.2,800/1.2)], multiscale_mode='value', keep_ratio=True),# bbox_clip_border=False),
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
        # img_scale=[(1333, 800), (1333*1.5, 800*1.5), (1333 * 2, 800 * 2), (1333*2.5, 800 *2.5), (1333*3, 800*3), (1333*3.5, 800*3.5)],
        # img_scale=[(1333, 800), (1333*1.5, 800*1.5), (1333 * 2, 800 * 2), (1333*2.5, 800 *2.5), (1333*3, 800*3)],
        # img_scale=[(1333, 800), (1333*1.5, 800*1.5)],
        img_scale=(1333, 800),
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco-cut_train.json',
        img_prefix=data_root + 'images/VisDrone2019-DET-train/images-cut',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/coco-val.json',
        img_prefix=data_root + 'images/VisDrone2019-DET-val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/coco-cut_test-dev.json',
        # ann_file='/home/sugar/workspace/dyh/coco-mod10-test-dev-iscrowd.json',
        ann_file=data_root + 'annotations/coco-test-dev.json',
        # img_prefix=data_root + 'images/VisDrone2019-DET-train/images',
        # img_prefix='/home/sugar/workspace/dyh//sequences',
        img_prefix=data_root + 'images/VisDrone2019-DET-test-dev/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
