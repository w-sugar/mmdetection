import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS
import torch
from ..losses import SmoothL1Loss
import matplotlib.pyplot as plt

def gaussian2D(radius_x, radius_y, sigma_x=1, sigma_y=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius_x, radius_x + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius_y, radius_y + 1, dtype=dtype, device=device).view(-1, 1)

    # h = (-(x * x + y * y) / (2 * sigma_x * sigma_y)).exp()
    h = (-((x * x / (2 * sigma_x * sigma_x)) + (y * y / (2 * sigma_y * sigma_y)))).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    radius_x = int(radius_x)
    radius_y = int(radius_y)
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1

    gaussian_kernel = gaussian2D(
        radius_x, radius_y, sigma_x=diameter_x / 6, sigma_y=diameter_y / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center
    x = int(x)
    y = int(y)

    height, width = heatmap.shape[:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap

@NECKS.register_module()
class BFP(BaseModule):
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
                 with_mask_loss=False,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(BFP, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local']

        self.with_mask_loss = with_mask_loss
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
        if self.with_mask_loss:
            self.maskconv1 = ConvModule(
                self.in_channels,
                self.in_channels // 4,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            self.maskconv2 = ConvModule(
                self.in_channels // 4,
                self.in_channels // 16,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            self.maskconv3 = ConvModule(
                self.in_channels // 16,
                1,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=dict(type='BN', requires_grad=True)
            )

            self.upsamplev1 = ConvModule(
                1, 
                16, 
                1,
                padding=0,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            self.upsamplev2 = ConvModule(
                16, 
                64, 
                1,
                padding=0,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
        # self.loss_mask = SmoothL1Loss(beta=1.0 / 9.0, loss_weight=1.0)
        self.loss_mask = torch.nn.MSELoss()

    def forward(self, inputs):
        inputs, gt_bboxes = inputs
        """Forward function."""
        assert len(inputs) == self.num_levels

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
            feats.append(gathered)
            # feats.append(gathered * 1 / (i + 1))

        bsf = sum(feats) / len(feats)
        # bsf = sum(feats)

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

        # 对fpn所有层进行mask监督
        if self.with_mask_loss:
            outs_upsample = []
            loss_mask = []
            for i in range(self.num_levels):
                out = outs[i]
                mask1 = self.maskconv1(out)
                mask2 = self.maskconv2(mask1)
                mask3 = self.maskconv3(mask2)
                # plt.imshow(mask3.squeeze(1)[0].cpu().detach().numpy())
                # plt.savefig('3.jpg')
                # a = input("aaaa")
                # mask = F.interpolate(mask3, size=[mask_size[0] * 4, mask_size[1] * 4], mode='nearest')
                if gt_bboxes is not None:
                    mask_size = out.size()[2:]
                    heatmaps = []
                    # x = 0
                    for gt_bbox in gt_bboxes:
                        heatmap = torch.zeros([mask_size[0], mask_size[1]], device=gt_bboxes[0].device)
                        # center = (gt_bbox / 16)
                        center = gt_bbox / (2 ** (i + 2))
                        Ws = center[:, 2] - center[:, 0]
                        Hs = center[:, 3] - center[:, 1]
                        center = center[:, :2] + (center[:, 2:] - center[:, :2]) / 2
                        center = torch.clamp(center, 0)
                        for cen, w, h in zip(center, Ws, Hs):
                            heatmap = gen_gaussian_target(heatmap, cen, w/2, h/2)
                        heatmaps.append(heatmap)
                        # plt.imshow(heatmap.cpu().numpy())
                        # plt.savefig(str(x)+'.jpg')
                    heatmaps = torch.stack(heatmaps)
                    loss_mask.append(self.loss_mask(mask3.squeeze(1), heatmaps))
                upsample1 = self.upsamplev1(mask3)
                upsample2 = self.upsamplev2(upsample1)
                outs_upsample.append(upsample2)
                # outs[i] = torch.cat([out, upsample2], dim=1)
            loss_mask = sum(loss_mask)

        if self.with_mask_loss and gt_bboxes is not None:
            return tuple(outs), tuple(outs_upsample), dict(loss_mask=loss_mask)
        else:
            return tuple(outs), tuple(outs_upsample), None
