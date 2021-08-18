checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/sugar/workspace/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# load_from = None
# resume_from = '/data/sugar/checkpoints/mmdetection_work_dirs/cascade_rcnn_r50_fpn_1x_coco_cut_four_multiscale_dcn_softroi_mosaic/latest.pth'
resume_from = None
workflow = [('train', 1)]
