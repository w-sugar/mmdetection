import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class TailStandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 bbox_head_tail=None,
                 labels=None,
                 labels_tail=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None
        super(TailStandardRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # if bbox_head is not None:
        self.init_bbox_head(bbox_roi_extractor, bbox_head, bbox_head_tail)
        self.labels = labels
        self.labels_tail = labels_tail
        self.init_assigner_sampler()

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        self.bbox_sampler_tail = None
        self.bbox_assigner_tail = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_assigner_tail = build_assigner(self.train_cfg.assigner_tail)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
            self.bbox_sampler_tail = build_sampler(
                self.train_cfg.sampler_tail, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head, bbox_head_tail):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head_tail = build_head(bbox_head_tail)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

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
        # assign gts and sample proposals
        proposal_list_tail = proposal_list.copy()
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_results_tail = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                assign_result_tail = self.bbox_assigner_tail.assign(
                    proposal_list_tail[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_tail = self.bbox_sampler_tail.sample(
                    assign_result_tail,
                    proposal_list_tail[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results_tail.append(sampling_result_tail)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results, bbox_results_tail = self._bbox_forward_train(x, sampling_results, sampling_results_tail,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            # losses.update(bbox_results['loss_bbox'])
            # losses.update(bbox_results_tail['loss_bbox'])
            for name, value in bbox_results['loss_bbox'].items():
                losses[f'{name}'] = (value if 'loss' in name else value)
            for name, value in bbox_results_tail['loss_bbox'].items():
                losses[f'tail.{name}'] = (value if 'loss' in name else value)
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    
    def _bbox_forward_tail(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head_tail(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, sampling_results_tail, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)

        rois_tail = bbox2roi([res.bboxes for res in sampling_results_tail])
        bbox_results_tail = self._bbox_forward_tail(x, rois_tail)

        bbox_targets = self.bbox_head_tail.get_targets(sampling_results_tail, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head_tail.loss(bbox_results_tail['cls_score'],
                                        bbox_results_tail['bbox_pred'], rois_tail,
                                        *bbox_targets)

        bbox_results_tail.update(loss_bbox=loss_bbox)
        return bbox_results, bbox_results_tail

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor or List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # The length of proposals of different batches may be different.
        # In order to form a batch, a padding operation is required.
        if isinstance(proposals, list):
            # padding to form a batch
            max_size = max([proposal.size(0) for proposal in proposals])
            for i, proposal in enumerate(proposals):
                supplement = proposal.new_full(
                    (max_size - proposal.size(0), proposal.size(1)), 0)
                proposals[i] = torch.cat((supplement, proposal), dim=0)
            rois = torch.stack(proposals, dim=0)
        else:
            rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        if not torch.onnx.is_in_onnx_export():
            # remove padding, ignore batch_index when calculating mask
            supplement_mask = rois.abs()[..., 1:].sum(dim=-1) == 0
            cls_score[supplement_mask, :] = 0

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.reshape(batch_size,
                                              num_proposals_per_img,
                                              bbox_pred.size(-1))
                if not torch.onnx.is_in_onnx_export():
                    bbox_pred[supplement_mask, :] = 0
            else:
                # TODO: Looking forward to a better way
                # For SABL
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                # apply bbox post-processing to each image individually
                det_bboxes = []
                det_labels = []
                for i in range(len(proposals)):
                    # remove padding
                    supplement_mask = proposals[i].abs().sum(dim=-1) == 0
                    for bbox in bbox_preds[i]:
                        bbox[supplement_mask] = 0
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_preds[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)
                return det_bboxes, det_labels
        else:
            bbox_pred = None

        return self.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shapes,
            scale_factors,
            rescale=rescale,
            cfg=rcnn_test_cfg)

    def simple_test_bboxes_tail(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor or List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # The length of proposals of different batches may be different.
        # In order to form a batch, a padding operation is required.
        if isinstance(proposals, list):
            # padding to form a batch
            max_size = max([proposal.size(0) for proposal in proposals])
            for i, proposal in enumerate(proposals):
                supplement = proposal.new_full(
                    (max_size - proposal.size(0), proposal.size(1)), 0)
                proposals[i] = torch.cat((supplement, proposal), dim=0)
            rois = torch.stack(proposals, dim=0)
        else:
            rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward_tail(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        if not torch.onnx.is_in_onnx_export():
            # remove padding, ignore batch_index when calculating mask
            supplement_mask = rois.abs()[..., 1:].sum(dim=-1) == 0
            cls_score[supplement_mask, :] = 0

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.reshape(batch_size,
                                              num_proposals_per_img,
                                              bbox_pred.size(-1))
                if not torch.onnx.is_in_onnx_export():
                    bbox_pred[supplement_mask, :] = 0
            else:
                # TODO: Looking forward to a better way
                # For SABL
                bbox_preds = self.bbox_head_tail.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                # apply bbox post-processing to each image individually
                det_bboxes = []
                det_labels = []
                for i in range(len(proposals)):
                    # remove padding
                    supplement_mask = proposals[i].abs().sum(dim=-1) == 0
                    for bbox in bbox_preds[i]:
                        bbox[supplement_mask] = 0
                    det_bbox, det_label = self.bbox_head_tail.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_preds[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale=rescale,
                        cfg=rcnn_test_cfg)
                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)
                return det_bboxes, det_labels
        else:
            bbox_pred = None

        return self.bbox_head_tail.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shapes,
            scale_factors,
            rescale=rescale,
            cfg=rcnn_test_cfg)

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        det_bboxes_tail, det_labels_tail = self.simple_test_bboxes_tail(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            return det_bboxes, det_labels

        if self.labels is not None:
            det_bboxes_post = []
            det_labels_post = []
            for i in range(num_imgs):
                inds = []
                for label in self.labels:
                    inds.append(torch.nonzero(det_labels[i] == label, as_tuple=False).squeeze(1))
                inds = torch.cat(inds)
                det_bboxes_post.append(det_bboxes[i][inds])
                det_labels_post.append(det_labels[i][inds])
        if self.labels_tail is not None:
            det_bboxes_tail_post = []
            det_labels_tail_post = []
            for i in range(num_imgs):
                inds = []
                for label in self.labels_tail:
                    inds.append(torch.nonzero(det_labels_tail[i] == label, as_tuple=False).squeeze(1))
                inds = torch.cat(inds)
                det_bboxes_tail_post.append(det_bboxes_tail[i][inds])
                det_labels_tail_post.append(det_labels_tail[i][inds])
        
        bbox_results = []
        for i in range(len(det_bboxes)):
            if det_bboxes_post[i].shape[0] == 0:
                det_bboxes_post[i] = torch.zeros([0, 5]).to(device=det_bboxes_post[i].device)
            if det_bboxes_tail_post[i].shape[0] == 0:
                det_bboxes_tail_post[i] = torch.zeros([0, 5]).to(device=det_bboxes_tail_post[i].device)
            assert det_bboxes_post[i].shape[1] == det_bboxes_tail_post[i].shape[1], (det_bboxes_post[i].shape, det_bboxes_tail_post[i].shape)
            det_bboxes_ = torch.cat((det_bboxes_post[i], det_bboxes_tail_post[i]))
            det_labels_ = torch.cat((det_labels_post[i], det_labels_tail_post[i]))
            bbox_result = bbox2result(det_bboxes_, det_labels_,
                        self.bbox_head.num_classes)

            bbox_results.append(bbox_result)
        # bbox_results = [
        #     bbox2result(det_bboxes[i], det_labels[i],
        #                 self.bbox_head.num_classes)
        #     for i in range(len(det_bboxes))
        # ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
