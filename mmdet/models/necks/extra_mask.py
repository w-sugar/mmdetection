import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS
import torch
from torch import nn
from ..losses import SmoothL1Loss
from ..losses import FocalLoss
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

def gen_rect_target(heatmap, center, radius_x, radius_y, k=1):
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

    x, y = center
    x = int(x)
    y = int(y)

    height, width = heatmap.shape[:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_rect = torch.ones(bottom+top, right+left, device=heatmap.device).to(torch.long)
    masked_rect = torch.ones(bottom+top, right+left, device=heatmap.device)
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_rect * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap

@NECKS.register_module()
class ExtraMask(BaseModule):

    def __init__(self,
                 in_channels,
                 num_levels,
                 with_mask_pooling=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ExtraMask, self).__init__(init_cfg)

        self.with_mask_pooling = with_mask_pooling
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

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

        # 不用上采样，直接用下采样之前的feature
        # self.upsamplev1 = ConvModule(
        #     1, 
        #     16, 
        #     1,
        #     padding=0,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        # self.upsamplev2 = ConvModule(
        #     16, 
        #     64, 
        #     1,
        #     padding=0,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        if with_mask_pooling:
            self.maskROIConv = nn.Sequential(
                ConvModule(
                    64, 
                    16, 
                    1,
                    padding=0,
                    norm_cfg=dict(type='BN', requires_grad=True)),
                ConvModule(
                    16, 
                    16, 
                    3,
                    padding=2,
                    stride=1,
                    dilation=2,
                    norm_cfg=dict(type='BN', requires_grad=True)),
                ConvModule(
                    16, 
                    64, 
                    1,
                    padding=0,
                    norm_cfg=dict(type='BN', requires_grad=True))
            )
            self.mask_upsample = ConvModule(
                                    64, 
                                    256, 
                                    1,
                                    padding=0,
                                    norm_cfg=dict(type='BN', requires_grad=True))
        # self.loss_mask = SmoothL1Loss(beta=1.0 / 9.0, loss_weight=1.0)
        self.loss_mask = torch.nn.MSELoss()
        # self.loss_mask = FocalLoss()

    def forward(self, inputs):
        inputs, gt_bboxes = inputs
        """Forward function."""
        assert len(inputs) == self.num_levels

        # 对fpn所有层进行mask监督
        outs_upsample = []
        loss_mask = []
        for i in range(self.num_levels):
            out = inputs[i]
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
                    # heatmap = torch.zeros([mask_size[0], mask_size[1]], device=gt_bboxes[0].device).to(torch.long)
                    heatmap = torch.zeros([mask_size[0], mask_size[1]], device=gt_bboxes[0].device)
                    # center = (gt_bbox / 16)
                    center = gt_bbox / (2 ** (i + 2))
                    Ws = center[:, 2] - center[:, 0]
                    Hs = center[:, 3] - center[:, 1]
                    center = center[:, :2] + (center[:, 2:] - center[:, :2]) / 2
                    center = torch.clamp(center, 0)
                    for cen, w, h in zip(center, Ws, Hs):
                        # heatmap = gen_gaussian_target(heatmap, cen, w/2, h/2)
                        # heatmap = gen_gaussian_target(heatmap, cen, w/4, h/4)
                        heatmap = gen_rect_target(heatmap, cen, w/2, h/2)
                    heatmaps.append(heatmap)
                    # plt.imshow(heatmap.cpu().numpy())
                    # plt.savefig(str(x)+'.jpg')
                # heatmaps = torch.stack(heatmaps).flatten(0)
                heatmaps = torch.stack(heatmaps)
                # loss_mask.append(self.loss_mask(mask3.squeeze(1).flatten(0).unsqueeze(1), heatmaps))
                loss_mask.append(self.loss_mask(mask3.squeeze(1), heatmaps))
            # upsample1 = self.upsamplev1(mask3)
            # upsample2 = self.upsamplev2(upsample1)
            if self.with_mask_pooling:
                maskROI = self.maskROIConv(mask1) + mask1
                maskROI = self.mask_upsample(maskROI)
                outs_upsample.append(maskROI)
            # outs[i] = torch.cat([out, upsample2], dim=1)
        loss_mask = sum(loss_mask)

        if self.with_mask_pooling:
            if gt_bboxes is not None:
                return inputs, tuple(outs_upsample), dict(loss_mask=loss_mask)
            else:
                return inputs, tuple(outs_upsample), None
        else:
            return inputs, None, None
