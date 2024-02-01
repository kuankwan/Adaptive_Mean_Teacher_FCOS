import argparse
import os

import numpy as np
import cv2
import torch

import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import build_config
from dataset.transforms import ValTransforms
from models.detector import build_model
from utils.misc import load_weight


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # basic
    parser.add_argument('--min_size', default=600, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1300, type=int,
                        help='the min size of input image')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use cuda')
    # model
    parser.add_argument('-v', '--version', default='fcos50', type=str,
                        help='build YOLOF')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    # dataset
    parser.add_argument('--root', default='D:\\datasets',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco-val',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 6
        data_dir = os.path.join(args.root, 'comic')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 8
        data_dir = os.path.join(args.root, 'foggy_cityscapes')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 8
        data_dir = os.path.join(args.root, 'foggy_cityscapes')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)
    # 1.定义模型结构，选取要可视化的层
    # config
    cfg = build_config(args)


    # build model
    model = build_model(args=args,
                        cfg=cfg,
                        device=device,
                        num_classes=num_classes,
                        trainable=False)
    # load trained weight
    model = load_weight(device=device,
                        model=model,
                        path_to_ckpt=args.weight)
    model.eval()
    for name, module in model._modules.items():
        print (name," : ",module)
    traget_layers = [model.fpn.input_projs[0]]

    # 2.读取图片，将图片转为RGB
    origin_img = cv2.imread('D:\\datasets\\foggy_cityscapes\\images\\val\\frankfurt_000001_012870_leftImg8bit_foggy_beta_0.02.png')
    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    # 3.图片预处理：resize、裁剪、归一化
    transform = ValTransforms(min_size=cfg['test_min_size'],
                              max_size=cfg['test_max_size'],
                              pixel_mean=cfg['pixel_mean'],
                              pixel_std=cfg['pixel_std'],
                              format=cfg['format'],
                              padding=cfg['val_padding'])
    x = transform(origin_img)[0]
    net_input = x.unsqueeze(0).to(device)


    # 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
    canvas_img = (x*255).byte().numpy().transpose(1, 2, 0)
    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

    # 5.实例化cam
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=traget_layers, use_cuda=False)
    grayscale_cam = cam(net_input)
    grayscale_cam = grayscale_cam[0, :]

    # 6.将feature map与原图叠加并可视化
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)