from mmdet.apis import init_detector, inference_detector

config_file = 'configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'checkpoints/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth'
checkpoint_file = '/data/sugar/checkpoints/mmdetection_work_dirs/cascade_rcnn_r50_fpn_1x_coco_cut_four_dcn_multiscale/latest.pth'
device = 'cuda:4'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
result = inference_detector(model, '/home/sugar/workspace/data/9999963_00000_d_0000063/9999963_00000_d_0000063_1_0.jpg')
print(result[0])