from ..builder import BACKBONES
from .resnet3d import ResNet3d
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, build_norm_layer, constant_init, kaiming_init, ConvModule
from torch.nn.modules.utils import _triple

@BACKBONES.register_module()
class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.
    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(self, frame_number=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'
        self.lateral_convs = nn.ModuleList()
        if frame_number == 4:
            in_channels = [64, 128]
            temporal_strides = [4, 2]
            self.conv_num = 2
        else:
            in_channels = [64, 128, 256]
            temporal_strides = [8, 4, 2]
            self.conv_num = 3
        for i in range(self.conv_num):
            l_conv = ConvModule(
                in_channels[i],
                in_channels[i],
                (temporal_strides[i], 1, 1),
                stride=1,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True))
            self.lateral_convs.append(l_conv)


    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        out = []
        for idx, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x)
            out.append(x)
        
        for idx, feat in enumerate(out):
            if idx < self.conv_num:
                feat = self.lateral_convs[idx](feat)
            feat = feat.squeeze(2)
            out[idx] = feat

        return out

@CONV_LAYERS.register_module()
class Conv2plus1d(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.
    https://arxiv.org/pdf/1711.11248.pdf.
    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int | tuple[int]): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int | tuple[int]): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN3d')):
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False

        # The middle-plane is calculated according to:
        # M_i = \floor{\frac{t * d^2 N_i-1 * N_i}
        #   {d^2 * N_i-1 + t * N_i}}
        # where d, t are spatial and temporal kernel, and
        # N_i, N_i-1 are planes
        # and inplanes. https://arxiv.org/pdf/1711.11248.pdf
        mid_channels = 3 * (
            in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= (
            in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)

        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=bias)

        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu(x)
        x = self.conv_t(x)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)