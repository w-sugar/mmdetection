import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .convfc_bbox_head import Shared2FCBBoxHead
import numpy as np

@HEADS.register_module()
class GSBBoxHeadWith(Shared2FCBBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self, gs_config=None, *args, **kwargs):
        super(GSBBoxHeadWith, self).__init__(*args,**kwargs)

        self.fc_cls = nn.Linear(self.cls_last_dim,
                                self.num_classes + 5)
        # self.label2binlabel = [torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        #                         torch.tensor([0, 1, 5, 2, 3, 5, 5, 5, 5, 4, 5, 5]),
        #                         torch.tensor([6, 6, 0, 6, 6, 1, 2, 3, 4, 6, 5, 6])]
        self.label2binlabel = [torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                                torch.tensor([4, 4, 4, 4, 4, 4, 0, 1, 2, 4, 3, 4]),
                                torch.tensor([5, 0, 1, 5, 2, 3, 5, 5, 5, 4, 5, 5]),
                                torch.tensor([0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2])]
        # self.pred_slice = [
        #     [0, 2],
        #     [2, 6],
        #     [8, 7],
        # ]
        self.pred_slice = [
            [0, 2],
            [2, 5],
            [7, 6],
            [13, 3],
        ]
        # self.fg_splits = [
        #     torch.tensor([0, 1, 3, 4, 9]),
        #     torch.tensor([2, 5, 6, 7, 8, 10])
        # ]
        self.fg_splits = [
            torch.tensor([6, 7, 8, 10]),
            torch.tensor([1, 2, 4, 5, 9]),
            torch.tensor([0, 3])
        ]
        self.others_sample_ratio = 3
        self.loss_bins = []
        for i in range(4):
            self.loss_bins.append(build_loss(gs_config.loss_bin))

    def _remap_labels(self, labels):

        num_bins = 4
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i].to(device=labels.device)
            new_bin_label = mapping[labels]

            if i < 1:
                weight = torch.ones_like(new_bin_label)
                # weight = torch.zeros_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label, self.label2binlabel[i][-1])
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _sample_others(self, label, length):

        # only works for non bg-fg bins

        fg = torch.where(label < length, torch.ones_like(label),
                         torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.others_sample_ratio)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num, ), replace=False)
            sample_idx = torch.from_numpy(sample_idx).cuda()
            fg[sample_idx] = 1
            weight = fg

        return weight

    def _slice_preds(self, cls_score):

        new_preds = []

        num_bins = 4
        for i in range(num_bins):
            start = self.pred_slice[i][0]
            length = self.pred_slice[i][1]
            sliced_pred = cls_score.narrow(1, start, length)
            new_preds.append(sliced_pred)

        return new_preds

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        if cls_score is not None:
            # Original label_weights is 1 for each roi.
            new_labels, new_weights, new_avgfactors = self._remap_labels(labels)
            new_preds = self._slice_preds(cls_score)

            num_bins = len(new_labels)
            for i in range(num_bins):
                losses['loss_cls_bin{}'.format(i)] = self.loss_bins[i](
                    new_preds[i],
                    new_labels[i],
                    new_weights[i],
                    avg_factor=new_avgfactors[i],
                    reduction_override=reduction_override
                )

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses


    @force_fp32(apply_to=('cls_score'))
    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. Decay the score of normal classes
        with the score of fg.
        From v1.
        '''

        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, self.num_classes + 1)).cuda()
        merge = torch.zeros((num_proposals, self.num_classes + 1)).cuda()

        # import pdb
        # pdb.set_trace()
        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i][:, :-1]

        weight = bg_score.narrow(1, 0, 1)

        # Whether we should add this? Test
        fg_merge = weight * fg_merge

        merge[:, -1] = bg_score[:, 1]
        merge[:, :-1] = fg_merge[:, :-1]
        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = self._merge_score(cls_score)

        batch_mode = True
        if rois.ndim == 2:
            # e.g. AugTest, Cascade R-CNN, HTC, SCNet...
            batch_mode = False

            # add batch dimension
            if scores is not None:
                scores = scores.unsqueeze(0)
            if bbox_pred is not None:
                bbox_pred = bbox_pred.unsqueeze(0)
            rois = rois.unsqueeze(0)

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        if rescale and bboxes.size(-2) > 0:
            if not isinstance(scale_factor, tuple):
                scale_factor = tuple([scale_factor])
            # B, 1, bboxes.size(-1)
            scale_factor = bboxes.new_tensor(scale_factor).unsqueeze(1).repeat(
                1, 1,
                bboxes.size(-1) // 4)
            bboxes /= scale_factor

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_size = scores.shape[0]
            # ignore background class
            scores = scores[..., :self.num_classes]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', cfg.max_per_img)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
        det_bboxes = []
        det_labels = []
        for (bbox, score) in zip(bboxes, scores):
            if cfg is not None:
                det_bbox, det_label = multiclass_nms(bbox, score,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
            else:
                det_bbox, det_label = bbox, score
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if not batch_mode:
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
        return det_bboxes, det_labels
