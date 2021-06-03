import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS
from torch import nn
from .test import _NonLocalNd
from mmcv.runner import auto_fp16
from mmcv.cnn import normal_init
import torch

@NECKS.register_module()
class WJF_P(BaseModule):
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
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(WJF_P, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        
        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

        self.Pconv1 = PConvMoule()
        self.Pconv2 = PConvMoule()
        self.Pconv3 = PConvMoule()

        # self.self_attn1 = nn.MultiheadAttention(256, 4, dropout=0.1)
        # self.self_attn1 = _NonLocalNd(256)
        # self.dropout1 = nn.Dropout(0.1)
        # self.norm1 = nn.LayerNorm(256)

        #self.self_attn2 = _NonLocalNd(256)
        #self.dropout2 = nn.Dropout(0.1)
        #self.norm2 = nn.LayerNorm(256)

       # self.self_attn3 = _NonLocalNd(256)
        #self.dropout3 = nn.Dropout(0.1)
        #self.norm3 = nn.LayerNorm(256)

        # self.self_attn2 = nn.MultiheadAttention(256, 8, dropout=0.1)
        # self.dropout2 = nn.Dropout(0.1)
        # self.norm2 = nn.LayerNorm(256)

        # self.reduceChannelConv = ConvModule(
        #     self.in_channels,
        #     self.in_channels // 128,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg
        # )
        # self.addChannelConv = ConvModule(
        #     self.in_channels // 128,
        #     self.in_channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg
        # )

    # @auto_fp16()
    # def forward(self, inputs):
    #     """Forward function."""
    #     inputs, gt_bboxes = inputs
    #     assert len(inputs) == self.num_levels
    #     b, c, h, w = inputs[self.refine_level].shape

    #     # step 1: gather multi-level features by resize and average
    #     feats = []
    #     gather_size = inputs[self.refine_level].size()[2:]
    #     for i in range(self.num_levels):
    #         if i < self.refine_level:
    #             gathered = F.adaptive_max_pool2d(
    #                 inputs[i], output_size=gather_size)
    #         else:
    #             gathered = F.interpolate(
    #                 inputs[i], size=gather_size, mode='nearest')
    #         # gathered = self.reduceChannelConv(gathered)
    #         gathered = gathered.flatten(2).permute(2, 0, 1)
    #         feats.append(gathered)

    #     # bsf = sum(feats) / len(feats)

    #     # self_attention c2-c4
    #     #print(feats[0].shape)
    #     pro_features = self.self_attn1(feats[0], feats[2], value=feats[1])[0]
    #     pro_features = feats[1] + self.dropout1(pro_features)
    #     pro_features = self.norm1(pro_features)

    #     # self_attention c4-c6
    #     pro_features2 = self.self_attn1(feats[2], feats[4], value=feats[3])[0]
    #     pro_features2 = feats[3] + self.dropout1(pro_features2)
    #     pro_features2 = self.norm1(pro_features2)

    #     # self_attention c3-c5
    #     pro_features3 = self.self_attn1(pro_features, pro_features2, value=feats[2])[0]
    #     pro_features3 = feats[2] + self.dropout1(pro_features3)
    #     bsf = self.norm1(pro_features3)
    #     bsf = pro_features3
    #     # reshape
    #     bsf = bsf.permute(1, 2, 0).view(b, c, h, w)

    #     # bsf = self.addChannelConv(bsf)
    #     # step 3: scatter refined features to multi-levels by a residual path
    #     outs = []
    #     for i in range(self.num_levels):
    #         out_size = inputs[i].size()[2:]
    #         if i < self.refine_level:
    #             residual = F.interpolate(bsf, size=out_size, mode='nearest')
    #         else:
    #             residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
    #         outs.append(residual + inputs[i])
    #         # outs.append(residual * 1 / (i + 1) + inputs[i])

    #     # return tuple(outs)
    #     return tuple(outs), None

    def forward(self, inputs):
        """Forward function."""
        inputs, gt_bboxes = inputs
        assert len(inputs) == self.num_levels
        b, c, h, w = inputs[self.refine_level].shape

        # step 1: gather multi-level features by resize and average

        pro_features = self.Pconv1([inputs[4], inputs[3], inputs[2]])
        pro_features2 = self.Pconv2([inputs[2], inputs[1], inputs[0]])
        pro_features3 = self.Pconv3([pro_features, inputs[2], pro_features2])

        bsf = pro_features3

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

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
        return tuple(outs), None, None


class PConvMoule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
    ):
        super(PConvMoule, self).__init__()

        # self.Pconv = nn.ModuleList()
        # self.Pconv.append(
        #     ConvModule(
        #         in_channels,
        #         out_channels,
        #         kernel_size[0],
        #         stride=1,
        #         padding=1,
        #         norm_cfg=dict(type='BN', requires_grad=True)
        #     )
        # )
        # self.Pconv.append(
        #     ConvModule(
        #         in_channels,
        #         out_channels,
        #         kernel_size[1],
        #         stride=1,
        #         padding=1,
        #         norm_cfg=dict(type='BN', requires_grad=True)
        #     )
        # )
        # self.Pconv.append(
        #     ConvModule(
        #         in_channels,
        #         out_channels,
        #         kernel_size[1],
        #         stride=2,
        #         padding=1,
        #         norm_cfg=dict(type='BN', requires_grad=True)
        #     )
        # )

        # 增加通道注意力
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.action_p2_squeeze = nn.Conv2d(in_channels, in_channels // 16, kernel_size=(1,1), stride=(1,1), bias=False, padding=(0,0))
        # self.action_p2_conv1 = nn.Conv1d(in_channels // 16, in_channels // 16, kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        # self.action_p2_expand = nn.Conv2d(in_channels // 16, out_channels, kernel_size=(1,1), stride=(1,1), bias=False, padding=(0,0))
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.Tconv = ConvModule(
            in_channels, 
            out_channels,
            (3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', requires_grad=True)
        )
    def forward(self, x):
        assert len(x) == 3
        out_size = x[1].size()[2:]
        # # level-1
        # temp_fea = self.Pconv[0](x[0])
        # # temp_fea = F.interpolate(temp_fea, size=out_size, mode='nearest')
        
        # # # level-2
        # # temp_fea += self.Pconv[1](x[1])

        # # # level-3
        # # temp_fea += self.Pconv[2](x[2])

        # temp_fea_1 = F.interpolate(temp_fea, size=out_size, mode='nearest')
        
        # # level-2
        # temp_fea_2 = self.Pconv[1](x[1])

        # # level-3
        # temp_fea_3 = self.Pconv[2](x[2])
        x[0] = F.interpolate(x[0], size=out_size, mode='nearest')
        x[2] = F.adaptive_max_pool2d(x[2], output_size=out_size)

        '''
        bs = x[1].size()[0]
        temp_fea = torch.cat([temp_fea_1, temp_fea_2, temp_fea_3], 0)
        x_p2 = self.avg_pool(temp_fea)
        x_p2 = self.action_p2_squeeze(x_p2)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(bs, 3, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        x_p2 = self.sigmoid(x_p2)
        # x_p2 = x_shift * x_p2 + x_shift
        x_p2 = temp_fea * x_p2
        x_p2 = x_p2.view(bs, 3, 256, out_size[0], out_size[1]).contiguous()
        x_p2 = torch.sum(x_p2, 1)
        '''

        # temp_fea = torch.stack([temp_fea_1, temp_fea_2, temp_fea_3], 2)
        temp_fea = torch.stack(x, 2)
        temp_fea = self.Tconv(temp_fea).squeeze(2)
        return temp_fea
        # return x_p2