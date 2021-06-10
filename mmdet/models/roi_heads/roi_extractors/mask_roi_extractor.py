import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class MaskRoIExtractor(BaseRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(MaskRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        # self.spatial_attention_conv=nn.Sequential(nn.Conv2d(out_channels*2, out_channels, 1), nn.ReLU(), nn.Conv2d(out_channels,2,3, padding=1))
        self.channel_attention_conv=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(out_channels*2, out_channels, 1), nn.ReLU(), nn.Conv2d(out_channels, out_channels*2, 1))
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

    def roi_rescale(self, rois, num_levels, level):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        rois_lvls = torch.zeros(num_levels, rois.shape[0], 5)
        for i in range(num_levels):
            scale_factor = 2 ** (i - level)
            cx = (rois[:, 1] + rois[:, 3]) * 0.5
            cy = (rois[:, 2] + rois[:, 4]) * 0.5
            w = rois[:, 3] - rois[:, 1]
            h = rois[:, 4] - rois[:, 2]
            new_w = w * scale_factor
            new_h = h * scale_factor
            x1 = cx - new_w * 0.5
            x2 = cx + new_w * 0.5
            y1 = cy - new_h * 0.5
            y2 = cy + new_h * 0.5
            new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
            rois_lvls[i] = new_rois
        return rois_lvls

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
        
        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_x, roi_feats_mask = torch.split(feats[i], [256, 256], 1)
                '''
                # 使用sac对两个roi进行合并
                roi_feats_x_ = self.roi_layers[i](roi_feats_x.contiguous(), rois_)
                roi_feats_mask_ = self.roi_layers[i](roi_feats_mask.contiguous(), rois_)
                concat_roi_feats = torch.cat([roi_feats_x_, roi_feats_mask_], dim=1)
                spatial_attention_conv = F.sigmoid(self.spatial_attention_conv(concat_roi_feats))
                roi_feats[inds] = spatial_attention_conv[:, 0, None, :, :] * roi_feats_x_ + spatial_attention_conv[:, 1, None, :, :] * roi_feats_mask_
                '''

                # 使用sac对两个feature进行合并
                # spatial_attention_conv = F.sigmoid(self.spatial_attention_conv(feats[i]))
                # feats_post = spatial_attention_conv[:, 0, None, :, :] * roi_feats_x + spatial_attention_conv[:, 1, None, :, :] * roi_feats_mask
                # roi_feats[inds] = self.roi_layers[i](feats_post, rois_)

                # 使用cac对两个feature进行合并
                channel_attention_conv = F.sigmoid(self.channel_attention_conv(feats[i]))
                feats_post = channel_attention_conv * feats[i]
                feats_x, feats_mask = torch.split(feats_post, [256, 256], 1)
                roi_feats[inds] = self.roi_layers[i](feats_x + feats_mask, rois_)

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

        # concat_roi_feats = torch.cat(roi_feats_list, dim=1)
        # spatial_attention_map = self.spatial_attention_conv(concat_roi_feats)
        # for i in range(num_levels):
        #     roi_feats += (F.sigmoid(spatial_attention_map[:, i, None, :, :]) * roi_feats_list[i])
        return roi_feats