import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS
from torch import nn


@NECKS.register_module()
class WJF(BaseModule):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(WJF, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        assert 0 <= self.refine_level < self.num_levels
        
        self.self_attn1 = nn.MultiheadAttention(128, 4, dropout=0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(128)

        # self.self_attn2 = nn.MultiheadAttention(256, 8, dropout=0.1)
        # self.dropout2 = nn.Dropout(0.1)
        # self.norm2 = nn.LayerNorm(256)

        self.reduceChannelConv = ConvModule(
            self.in_channels,
            self.in_channels // 2,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )
        self.addChannelConv = ConvModule(
            self.in_channels // 2,
            self.in_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )

    def forward(self, inputs):
        """Forward function."""
        inputs, gt_bboxes = inputs
        assert len(inputs) == self.num_levels
        b, c, h, w = inputs[self.refine_level].shape

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            gathered = self.reduceChannelConv(gathered)
            gathered = gathered.flatten(2).permute(2, 0, 1)
            feats.append(gathered)

        # bsf = sum(feats) / len(feats)

        # self_attention c2-c4
        pro_features = self.self_attn1(feats[0], feats[2], value=feats[1])[0]
        pro_features = feats[1] + self.dropout1(pro_features)
        pro_features = self.norm1(pro_features)

        # self_attention c4-c6
        pro_features2 = self.self_attn1(feats[2], feats[4], value=feats[3])[0]
        pro_features2 = feats[3] + self.dropout1(pro_features2)
        pro_features2 = self.norm1(pro_features2)

        # self_attention c3-c5
        pro_features3 = self.self_attn1(pro_features, pro_features2, value=feats[2])[0]
        pro_features3 = feats[2] + self.dropout1(pro_features3)
        bsf = self.norm1(pro_features3)

        # reshape
        bsf = bsf.permute(1, 2, 0).view(b, c // 2, h, w)

        bsf = self.addChannelConv(bsf)
        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
            # outs.append(residual * 1 / (i + 1) + inputs[i])

        # return tuple(outs)
        return tuple(outs), None
