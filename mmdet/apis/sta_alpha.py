import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)



def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # runner.run(data_loaders, cfg.workflow)
    anchor_generator = build_anchor_generator(cfg.model.rpn_head.anchor_generator)
    assigner = build_assigner(cfg.model.train_cfg.rpn.assigner)
    total_num_targets = torch.tensor([0] * 5)
    for iteration, data in enumerate(data_loaders):
        for i in data:
            # print(i.keys())
            img_metas = i['img_metas']._data
            # print(img_metas)
            num_imgs = len(img_metas)
            images = i['img']._data
            gt_bboxes = i['gt_bboxes']._data
            h, w = images[0].size()[-2:]
            features_shape = []
            for i in range(2, 7):
                f_shape = [int(h/(2**i)), int(w/(2**i))]
                features_shape.append(f_shape)
            multi_level_anchors = anchor_generator.grid_anchors(
                features_shape)
            anchor_list = [multi_level_anchors for _ in range(num_imgs)]

            # for each image, we compute valid flags of multi level anchors
            valid_flag_list = []
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = anchor_generator.valid_flags(
                    features_shape, img_meta[0]['pad_shape'])
                valid_flag_list.append(multi_level_flags)
            # print(anchor_list, valid_flag_list)
            assert len(anchor_list) == len(valid_flag_list) == num_imgs

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors to a single tensor
            concat_anchor_list = []
            concat_valid_flag_list = []
            for i in range(num_imgs):
                assert len(anchor_list[i]) == len(valid_flag_list[i])
                concat_anchor_list.append(torch.cat(anchor_list[i]))
                concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
            gt_bboxes_ignore_list= None
            # compute targets for each image
            if gt_bboxes_ignore_list is None:
                gt_bboxes_ignore_list = [None for _ in range(num_imgs)]

            inside_flags = anchor_inside_flags(concat_anchor_list[0], concat_valid_flag_list[0],
                                           img_metas[0][0]['img_shape'][:2],
                                           0)
            if not inside_flags.any():
                return (None, ) * 7
            # assign gt and sample anchors
            anchors = concat_anchor_list[0][inside_flags, :]

            assign_result = assigner.assign(
                anchors.cpu(), gt_bboxes[0][0], gt_bboxes_ignore_list[0],
                None)
            # print(assign_result.pos_gt_bboxes)
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            labels = anchors.new_full((anchors.shape[0], ),
                                  -1,
                                  dtype=torch.long)
            labels[pos_inds] = 1
            num_total_anchors = concat_anchor_list[0].size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=-1)  # fill bg label
            match_results = images_to_levels([labels], num_level_anchors)
            # print(match_results)
            for idx, match_result in enumerate(match_results):
                num = torch.where(match_result==1)[0].numel()
                total_num_targets[idx] += num
            # print(total_num_targets)
        print(total_num_targets)
    print(total_num_targets)



            
