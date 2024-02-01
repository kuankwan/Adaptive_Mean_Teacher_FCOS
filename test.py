import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.voc import VOC_CLASSES, VOCDetection
from dataset.coco import COCODataset,coco_class_color
from dataset.transforms import ValTransforms
from models.detector.fcos.attfcos import T_SNE, draw
from utils.misc import load_weight, TestTimeAugmentation

from config import build_config
from models.detector import build_model
from utils.visualize_similarity_attention import heatmap


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')

    # basic
    parser.add_argument('--min_size', default=600, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1300, type=int,
                        help='the min size of input image')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--save_name', default='foggy', type=str,
                        help='Dir to save results')
    parser.add_argument('--test_score', default=0.5, type=float,
                        help='test score')
    parser.add_argument('--car_only', default=False,
                        help='test score')

    # model
    parser.add_argument('-v', '--version', default='fcos50', type=str,
                        help='build yolof')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='D:\\datasets',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()



def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names,
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    box_nums = 0
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            box_nums += 1
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) >= 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img,box_nums
        

def test(args,
         net, 
         device, 
         dataset,
         tar_dataset=None,
         transform=None,
         vis_thresh=0.5,
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         show=False,
         test_aug=None, 
         dataset_name='coco'):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version,args.save_name)
    os.makedirs(save_path, exist_ok=True)
    t = 0
    boxes = 0
    num_warmup = 5
    pure_inf_time = 0
    batch_sim = []
    tnse_target=[]
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape
        orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

        # prepare
        x = transform(image)[0]
        x = x.unsqueeze(0).to(device)


        t0 = time.time()
        # inference
        start_time = time.perf_counter()
        if test_aug is not None:
            # test augmentation:
            bboxes, scores, cls_inds,cos_sim = test_aug(x, net)
        else:
            bboxes, scores, cls_inds,cos_sim = net(x)
        elapsed = time.perf_counter() - start_time
        if index >= num_warmup:
            pure_inf_time += elapsed
        if (index + 1) == num_images:
            fps = (index + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
        t += time.time() - t0
        print("detection time used ", time.time() - t0, "s")
        #
        if cos_sim is not None:
            batch_sim.append(cos_sim)

        # rescale
        if transform.padding:
            # The input image is padded with 0 on the short side, aligning with the long side.
            bboxes *= max(orig_h, orig_w)
        else:
            # the input image is not padded.
            bboxes *= orig_size
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

        # vis detection
        img_processed,box_nums = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=dataset_name)
        boxes = box_nums + boxes
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        if isinstance(dataset,VOCDetection):
            cv2.imwrite(os.path.join(save_path, dataset.get_idx() +'.jpg'), img_processed)
        else:
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)
    if tar_dataset is not None:
        num_images = len(tar_dataset)
        for index in range(num_images):
            t_image, _ = tar_dataset.pull_image(index)
            tx = transform(t_image)[0]
            tx = tx.unsqueeze(0).to(device)
            if test_aug is not None:
                # test augmentation:
                _, _, _, cos_sim = test_aug(tx, net)
            else:
                _, _, _, cos_sim = net(tx)
            if cos_sim is not None:
                tnse_target.append(cos_sim)
    batch_sim = torch.cat(batch_sim,dim=0)
    tnse_target = torch.cat(tnse_target, dim=0)
    src_pred = T_SNE(np.array(batch_sim.cpu()))
    tgt_pred = T_SNE(np.array(tnse_target.cpu()))
    draw(src_pred, tgt_pred, 'aatest_tnse')
    # heatmap(batch_sim.sigmoid(),save_path='./visualization/heatmap/heat.jpg')
    print("FPS: ", num_images / t)
    print("Box_nums: ",boxes)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.car_only:
        coco_class_labels = ('car',)
        coco_class_color = [(248, 203, 127), ]
        coco_class_index = [0, ]
        num_classes=1
    else:
        coco_class_labels = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
        coco_class_color = [(99, 178, 238), (118, 218, 145), (248, 203, 127), (248, 149, 136), (124, 214, 207),
                        (145, 146, 171), (148, 60, 57), (98, 76, 124)]
        coco_class_index = [0, 1, 2, 3, 4, 5, 6, 7]
        num_classes=8
    # dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'comic')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 6
        dataset = VOCDetection(
                        data_dir=data_dir,
                        image_sets=[('2012', 'val')],
                        transform=None)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'cityscapes')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = num_classes
        dataset = COCODataset(
                    data_dir=data_dir,
                    image_set='car_val',
                    transform=None)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    tar_data_dir = os.path.join(args.root, 'foggy_cityscapes')
    tar_dataset = COCODataset(
                    data_dir=tar_data_dir,
                    image_set='val',
                    transform=None)

    np.random.seed(0)


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

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None

    # transform
    transform = ValTransforms(min_size=cfg['test_min_size'], 
                              max_size=cfg['test_max_size'],
                              pixel_mean=cfg['pixel_mean'],
                              pixel_std=cfg['pixel_std'],
                              format=cfg['format'],
                              padding=cfg['val_padding'])

    # run
    test(args=args,
        net=model, 
        device=device, 
        dataset=dataset,
        tar_dataset=tar_dataset,
        transform=transform,
        vis_thresh=args.test_score,
        class_colors=coco_class_color,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        test_aug=test_aug,
        dataset_name=args.dataset)
