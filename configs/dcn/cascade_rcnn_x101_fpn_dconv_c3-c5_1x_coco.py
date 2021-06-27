_base_ = '../cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py'
# model = dict(
#     pretrained='open-mmlab://resnext101_64x4d',
#     backbone=dict(
#         type='ResNeXt',
#         depth=101,
#         groups=64,
#         base_width=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
#         stage_with_dcn=(False, True, True, True),
#         style='pytorch'))
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnext50_32x4d_gn_ws',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg))