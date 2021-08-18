import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SoftRoIExtractor(BaseRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SoftRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        # self.spatial_attention_conv=nn.Sequential(nn.Conv2d(out_channels*len(featmap_strides), out_channels, 1), nn.ReLU(), nn.Conv2d(out_channels,len(featmap_strides),3, padding=1))
        self.self_attn1 = nn.MultiheadAttention(256, 2, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(256)
    
    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        
        roi_feats_list_v = []
        roi_feats_list_q = []
        roi_feats_list_k = []
        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                q_i = max(0, i - 1)
                k_i = min(num_levels-1, i+1)
                rois_ = rois[inds]
                roi_feats_v = self.roi_layers[i](feats[i], rois_)
                roi_feats_q = self.roi_layers[q_i](feats[q_i], rois_)
                roi_feats_k = self.roi_layers[k_i](feats[k_i], rois_)
                # 生成pos emb
                pos = self.PositionEmbeddingSine(roi_feats_v)
                pos = pos.flatten(2).permute(2, 0, 1)

                roi_feats_v = roi_feats_v.flatten(2).permute(2, 0, 1)
                roi_feats_q = roi_feats_q.flatten(2).permute(2, 0, 1)
                roi_feats_k = roi_feats_k.flatten(2).permute(2, 0, 1)
                # 取平均
                roi_feats_ = (roi_feats_v + roi_feats_q + roi_feats_k) / 3
                # 加入pos emb
                q = k = self.with_pos_embed(roi_feats_, pos)

                # pro_features = self.self_attn1(roi_feats_q, roi_feats_k, value=roi_feats_v)[0]
                # pro_features = roi_feats_v + self.dropout1(pro_features)
                pro_features = self.self_attn1(q, k, value=roi_feats_)[0]
                pro_features = roi_feats_v + self.dropout1(pro_features)

                pro_features = self.norm1(pro_features)
                pro_features = pro_features.permute(1, 2, 0).view(rois_.shape[0], 256, out_size[0], out_size[1])

                roi_feats[inds] = pro_features
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        return roi_feats
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def PositionEmbeddingSine(self, x):
        num_pos_feats=128
        temperature=10000
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device) > 0
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if True:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos