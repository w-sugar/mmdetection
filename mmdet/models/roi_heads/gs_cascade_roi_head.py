import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class GSCascadeRoIHead(CascadeRoIHead):
    """Cascade roi head including one bbox head and one mask head.
    https://arxiv.org/abs/1712.00726
    """

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner_bg = []
        self.bbox_sampler_bg = []
        self.bbox_assigner_human = []
        self.bbox_sampler_human = []
        self.bbox_assigner_vehicle = []
        self.bbox_sampler_vehicle = []
        self.bbox_assigner_two = []
        self.bbox_sampler_two = []
        self.bbox_assigner_three = []
        self.bbox_sampler_three = []
        if self.train_cfg is not None:
            for rcnn_train_cfg in self.train_cfg:
                self.bbox_assigner_bg.append(build_assigner(rcnn_train_cfg.assigner_bg))
                self.bbox_assigner_human.append(build_assigner(rcnn_train_cfg.assigner_human))
                self.bbox_assigner_vehicle.append(build_assigner(rcnn_train_cfg.assigner_vehicle))
                self.bbox_assigner_two.append(build_assigner(rcnn_train_cfg.assigner_two))
                self.bbox_assigner_three.append(build_assigner(rcnn_train_cfg.assigner_three))
                self.bbox_sampler_bg.append(build_sampler(rcnn_train_cfg.sampler_bg))
                self.bbox_sampler_human.append(build_sampler(rcnn_train_cfg.sampler_human))
                self.bbox_sampler_vehicle.append(build_sampler(rcnn_train_cfg.sampler_vehicle))
                self.bbox_sampler_two.append(build_sampler(rcnn_train_cfg.sampler_two))
                self.bbox_sampler_three.append(build_sampler(rcnn_train_cfg.sampler_three))


    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results_bg, sampling_results_human, sampling_results_vehicle, sampling_results_two, sampling_results_three, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois_bg = bbox2roi([res.bboxes for res in sampling_results_bg])
        bbox_results_bg = self._bbox_forward(stage, x, rois_bg)
        bbox_targets_bg = self.bbox_head[stage].get_targets(
            sampling_results_bg, gt_bboxes, gt_labels, rcnn_train_cfg)

        rois_human = bbox2roi([res.bboxes for res in sampling_results_human])
        bbox_results_human = self._bbox_forward(stage, x, rois_human)
        bbox_targets_human = self.bbox_head[stage].get_targets(
            sampling_results_human, gt_bboxes, gt_labels, rcnn_train_cfg)

        rois_vehicle = bbox2roi([res.bboxes for res in sampling_results_vehicle])
        bbox_results_vehicle = self._bbox_forward(stage, x, rois_vehicle)
        bbox_targets_vehicle = self.bbox_head[stage].get_targets(
            sampling_results_vehicle, gt_bboxes, gt_labels, rcnn_train_cfg)

        rois_two = bbox2roi([res.bboxes for res in sampling_results_two])
        bbox_results_two = self._bbox_forward(stage, x, rois_two)
        bbox_targets_two = self.bbox_head[stage].get_targets(
            sampling_results_two, gt_bboxes, gt_labels, rcnn_train_cfg)

        rois_three = bbox2roi([res.bboxes for res in sampling_results_three])
        bbox_results_three = self._bbox_forward(stage, x, rois_three)
        bbox_targets_three = self.bbox_head[stage].get_targets(
            sampling_results_three, gt_bboxes, gt_labels, rcnn_train_cfg)

        loss_bbox_bg = self.bbox_head[stage].loss(1, bbox_results_bg['cls_score'],
                                               bbox_results_bg['bbox_pred'], rois_bg,
                                               *bbox_targets_bg)
        bbox_results_bg.update(loss_bbox=loss_bbox_bg, rois=rois_bg, bbox_targets=bbox_targets_bg)

        loss_bbox_human = self.bbox_head[stage].loss(2, bbox_results_human['cls_score'],
                                               bbox_results_human['bbox_pred'], rois_human,
                                               *bbox_targets_human)
        bbox_results_human.update(loss_bbox=loss_bbox_human, rois=rois_human, bbox_targets=bbox_targets_human)

        loss_bbox_vehicle = self.bbox_head[stage].loss(3, bbox_results_vehicle['cls_score'],
                                               bbox_results_vehicle['bbox_pred'], rois_vehicle,
                                               *bbox_targets_vehicle)
        bbox_results_vehicle.update(loss_bbox=loss_bbox_vehicle, rois=rois_vehicle, bbox_targets=bbox_targets_vehicle)

        loss_bbox_two = self.bbox_head[stage].loss(4, bbox_results_two['cls_score'],
                                               bbox_results_two['bbox_pred'], rois_two,
                                               *bbox_targets_two)
        bbox_results_two.update(loss_bbox=loss_bbox_two, rois=rois_two, bbox_targets=bbox_targets_two)

        loss_bbox_three = self.bbox_head[stage].loss(5, bbox_results_three['cls_score'],
                                               bbox_results_three['bbox_pred'], rois_three,
                                               *bbox_targets_three)
        bbox_results_three.update(loss_bbox=loss_bbox_three, rois=rois_three, bbox_targets=bbox_targets_three)

        return bbox_results_bg, bbox_results_human, bbox_results_vehicle, bbox_results_two, bbox_results_three

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        # print(len(self.bbox_assigner))
        proposal_list_human = proposal_list.copy()
        proposal_list_vehicle = proposal_list.copy()
        proposal_list_two = proposal_list.copy()
        proposal_list_three = proposal_list.copy()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results_bg = []
            sampling_results_human = []
            sampling_results_vehicle = []
            sampling_results_two = []
            sampling_results_three = []
            if self.with_bbox or self.with_mask:
                bbox_assigner_bg = self.bbox_assigner_bg[i]
                bbox_assigner_human = self.bbox_assigner_human[i]
                bbox_assigner_vehicle = self.bbox_assigner_vehicle[i]
                bbox_assigner_two = self.bbox_assigner_two[i]
                bbox_assigner_three = self.bbox_assigner_three[i]
                bbox_sampler_bg = self.bbox_sampler_bg[i]
                bbox_sampler_human = self.bbox_sampler_human[i]
                bbox_sampler_vehicle = self.bbox_sampler_vehicle[i]
                bbox_sampler_two = self.bbox_sampler_two[i]
                bbox_sampler_three = self.bbox_sampler_three[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for j in range(num_imgs):
                    assign_result_bg = bbox_assigner_bg.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_bg = bbox_sampler_bg.sample(
                        assign_result_bg,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_bg.append(sampling_result_bg)
                    assign_result_human = bbox_assigner_human.assign(
                        proposal_list_human[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_human = bbox_sampler_human.sample(
                        assign_result_human,
                        proposal_list_human[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_human.append(sampling_result_human)
                    assign_result_vehicle = bbox_assigner_vehicle.assign(
                        proposal_list_vehicle[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_vehicle = bbox_sampler_vehicle.sample(
                        assign_result_vehicle,
                        proposal_list_vehicle[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_vehicle.append(sampling_result_vehicle)
                    assign_result_two = bbox_assigner_two.assign(
                        proposal_list_two[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_two = bbox_sampler_two.sample(
                        assign_result_two,
                        proposal_list_two[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_two.append(sampling_result_two)
                    assign_result_three = bbox_assigner_three.assign(
                        proposal_list_three[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result_three = bbox_sampler_three.sample(
                        assign_result_three,
                        proposal_list_three[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results_three.append(sampling_result_three)
            # bbox head forward and loss
            bbox_results_bg, bbox_results_human, bbox_results_vehicle, bbox_results_two, bbox_results_three = self._bbox_forward_train(i, x, sampling_results_bg, sampling_results_human, sampling_results_vehicle, sampling_results_two, sampling_results_three,
                                                                       gt_bboxes, gt_labels,
                                                                       rcnn_train_cfg)

            for name, value in bbox_results_bg['loss_bbox'].items():
                losses[f's{i}.bg.{name}'] = (value * lw if 'loss' in name else value)
            for name, value in bbox_results_human['loss_bbox'].items():
                losses[f's{i}.human.{name}'] = (value * lw if 'loss' in name else value)
            for name, value in bbox_results_vehicle['loss_bbox'].items():
                losses[f's{i}.vehicle.{name}'] = (value * lw if 'loss' in name else value)
            for name, value in bbox_results_two['loss_bbox'].items():
                losses[f's{i}.two.{name}'] = (value * lw if 'loss' in name else value)
            for name, value in bbox_results_three['loss_bbox'].items():
                losses[f's{i}.three.{name}'] = (value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results_bg]
                pos_is_gts_human = [res.pos_is_gt for res in sampling_results_human]
                pos_is_gts_vehicle = [res.pos_is_gt for res in sampling_results_vehicle]
                pos_is_gts_two = [res.pos_is_gt for res in sampling_results_two]
                pos_is_gts_three = [res.pos_is_gt for res in sampling_results_three]
                # bbox_targets is a tuple
                roi_labels = bbox_results_bg['bbox_targets'][0]
                roi_labels_human = bbox_results_human['bbox_targets'][0]
                roi_labels_vehicle = bbox_results_vehicle['bbox_targets'][0]
                roi_labels_two = bbox_results_two['bbox_targets'][0]
                roi_labels_three = bbox_results_three['bbox_targets'][0]

                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        bbox_results_bg['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results_bg['rois'], roi_labels,
                        bbox_results_bg['bbox_pred'], pos_is_gts, img_metas)
                    roi_labels_human = torch.where(
                        roi_labels_human == self.bbox_head[i].num_classes,
                        bbox_results_human['cls_score'][:, :-1].argmax(1),
                        roi_labels_human)
                    proposal_list_human = self.bbox_head[i].refine_bboxes(
                        bbox_results_human['rois'], roi_labels_human,
                        bbox_results_human['bbox_pred'], pos_is_gts_human, img_metas)
                    roi_labels_vehicle = torch.where(
                        roi_labels_vehicle == self.bbox_head[i].num_classes,
                        bbox_results_vehicle['cls_score'][:, :-1].argmax(1),
                        roi_labels_vehicle)
                    proposal_list_vehicle = self.bbox_head[i].refine_bboxes(
                        bbox_results_vehicle['rois'], roi_labels_vehicle,
                        bbox_results_vehicle['bbox_pred'], pos_is_gts_vehicle, img_metas)
                    roi_labels_two = torch.where(
                        roi_labels_two == self.bbox_head[i].num_classes,
                        bbox_results_two['cls_score'][:, :-1].argmax(1),
                        roi_labels_two)
                    proposal_list_two = self.bbox_head[i].refine_bboxes(
                        bbox_results_two['rois'], roi_labels_two,
                        bbox_results_two['bbox_pred'], pos_is_gts_two, img_metas)
                    roi_labels_three = torch.where(
                        roi_labels_three == self.bbox_head[i].num_classes,
                        bbox_results_three['cls_score'][:, :-1].argmax(1),
                        roi_labels_three)
                    proposal_list_three = self.bbox_head[i].refine_bboxes(
                        bbox_results_three['rois'], roi_labels_three,
                        bbox_results_three['bbox_pred'], pos_is_gts_three, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'][:, :-1].argmax(
                        dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[[]
                                for _ in range(self.mask_head[-1].num_classes)]
                               ]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]