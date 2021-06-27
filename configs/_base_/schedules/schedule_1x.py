# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# lr_config = dict(policy='step', step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
