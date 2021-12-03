model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_factors=[1.368, 0.727, 0.470],
        with_ExtraMask=[256,5,True,True],
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=False,
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='TailCascadeRoIHead',
        # labels_tail=[2, 5, 6, 7, 8, 10],
        # labels=[0, 1, 3, 4, 9],
        labels_tail=[2, 4, 5, 6, 7, 8, 9],
        labels=[0, 1, 3],
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            # type='SingleRoIExtractor',
            type='SoftRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        bbox_head_tail=[
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=2.0)),
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=2.0)),
            dict(
                # type='Shared2FCBBoxHead',
                type='GSBBoxHeadWith',
                gs_config=dict(
                    loss_bg=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bin=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                    ),
                ),
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0),
                # loss_cls=dict(
                #     type='WordTreeFocalLoss',
                #     use_sigmoid=True,
                #     gamma=2.0,
                #     alpha=0.25,
                #     loss_weight=2.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))
        ]
        ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[0, 1, 3, 4, 9],
                    labels=[0, 1, 3],
                    add_gt_as_proposals=True),
                assigner_tail=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler_tail=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[2, 5, 6, 7, 8, 10],
                    labels=[2, 4, 5, 6, 7, 8, 9],
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[0, 1, 3, 4, 9],
                    labels=[0, 1, 3],
                    add_gt_as_proposals=True),
                assigner_tail=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler_tail=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[2, 5, 6, 7, 8, 10],
                    labels=[2, 4, 5, 6, 7, 8, 9],
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[0, 1, 3, 4, 9],
                    labels=[0, 1, 3],
                    add_gt_as_proposals=True),
                assigner_tail=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=0.5),
                sampler_tail=dict(
                    type='ClassBalancedPosSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    # labels=[2, 5, 6, 7, 8, 10],
                    labels=[2, 4, 5, 6, 7, 8, 9],
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=500)))
dataset_type = 'VisDroneDataset'
data_root = 'data/visdrone/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(1600, 1050),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(3000, 1969), (3200, 2100), (3400, 2231)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, bbox_clip_border=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='VisDroneDataset',
        ann_file='data/visdrone/annotations/coco-cut_train_val_NOother_new.json',
        img_prefix='data/visdrone/images/VisDrone2019-DET-train_val/images-cut-NOother-new',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=(1600, 1050),
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='VisDroneDataset',
        ann_file='data/visdrone/annotations/coco-test-dev.json',
        img_prefix='data/visdrone/images/VisDrone2019-DET-test-dev/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', keep_ratio=True,
                        bbox_clip_border=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='VisDroneDataset',
        ann_file='data/visdrone/annotations/coco-test-dev.json',
        img_prefix='data/visdrone/images/VisDrone2019-DET-test-dev/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(3000, 1969), (3200, 2100), (3400, 2231)],
                flip=False,
                transforms=[
                    dict(
                        type='Resize', keep_ratio=True,
                        bbox_clip_border=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/sugar/workspace/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '/data/sugar/checkpoints/mmdetection_work_dirs/cascade_rcnn_r50_fpn_1x_coco_cut_four_multiscale_dcn_softroi/'
gpu_ids = range(0, 4)