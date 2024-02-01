from __future__ import division
from cmath import e
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.transforms import TrainTransforms, ValTransforms, BaseTransforms

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler
from utils.solver.warmup_schedule import build_warmup

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from config import build_config
from models.detector import build_model
from utils.logger import *
from tqdm import tqdm
from datetime import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size on single GPU')
    parser.add_argument('--schedule', type=str, default='1x',
                        help='training schedule.')
    parser.add_argument('--lr_scheduler', type=str, default='step',
                        help='lr scheduler.')
    parser.add_argument('-lr', '--base_lr', type=float, default=0.01,
                        help='base learning rate')
    parser.add_argument('-lr_bk', '--backbone_lr', type=float, default=0.01,
                        help='backbone learning rate')
    parser.add_argument('-lr_da', '--da_lr', type=float, default=0.01,
                        help='backbone learning rate')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='path to save weight')
    parser.add_argument('--eval_epoch', type=int,
                        default=1, help='interval between evaluations')

    # model
    parser.add_argument('-v', '--version', default='yolof18', type=str,
                        help='build yolof')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')

    # train trick
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='Mosaic augmentation')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False,
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version, date_time)
    os.makedirs(path_to_save, exist_ok=True)

    # logger
    global logger
    log_dir = os.path.join(path_to_save, 'log')
    logger = get_logger('train', log_dir)
    gpus_type, gpus_num = torch.cuda.get_device_name(
    ), torch.cuda.device_count()
    log_info = f'gpus_type: {gpus_type}, gpus_num: {gpus_num}'
    logger.info(log_info)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda:6")
    else:
        device = torch.device("cpu")

    # config
    cfg = build_config(args)
    for key, value in cfg.items():
        log_info = f'{key}: {value}'
        logger.info(log_info)

    # dataset and evaluator
    dataset, tar_dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, CollateFunc())
    tar_dataloader = build_dataloader(args, tar_dataset, CollateFunc())

    # build model
    net = build_model(
        args=args,
        cfg=cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        coco_pretrained=args.coco_pretrained,
        resume=args.resume)
    model = net
    model = model.to(device).train()
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info)

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(model=model_copy,
                         min_size=cfg['test_min_size'],
                         max_size=cfg['test_max_size'],
                         device=device)
        model_copy.trainable = True
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # optimizer
    optimizer, start_epoch = build_optimizer(
        model=model_without_ddp,
        base_lr=args.base_lr,
        backbone_lr=args.backbone_lr,
        da_lr=args.da_lr,
        name=cfg['optimizer'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        resume=args.resume
    )

    # lr scheduler
    lr_scheduler = build_lr_scheduler(
        cfg=cfg,
        args=args,
        name=args.lr_scheduler,
        optimizer=optimizer,
        resume=args.resume
    )

    # warmup scheduler
    warmup_scheduler = build_warmup(
        name=cfg['warmup'],
        base_lr=args.base_lr,
        wp_iter=cfg['wp_iter'],
        warmup_factor=cfg['warmup_factor']
    )

    # tensorboard
    path_to_write = os.path.join(path_to_save, 'log_loss')
    os.makedirs(path_to_write, exist_ok=True)
    trainer_writer = SummaryWriter(log_dir=path_to_write)

    # training configuration
    max_epoch = cfg['epoch'][args.schedule]['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    # print("best_map:{}".format(best_map))
    warmup = not args.no_warmup
    t0 = time.time()
    bar = tqdm(range(max_epoch))
    train_iter = iter(dataloader)
    tar_train_iter = iter(tar_dataloader)
    # start training loop
    for step, num_iter in enumerate(bar):
        visual = False
        # train one epoch
        # for iter_i, (images, targets, masks) in enumerate(dataloader):
        ni = num_iter
        # warmup
        if ni < cfg['wp_iter'] and warmup:
            warmup_scheduler.warmup(ni, optimizer)

        elif ni == cfg['wp_iter'] and warmup:
            # warmup is over
            print('Warmup is over')
            warmup = False
            warmup_scheduler.set_lr(optimizer, args.base_lr, args.base_lr)
        # data
        try:
            (images, targets, masks) = next(train_iter)
        except:
            train_iter = iter(dataloader)
            (images, targets, masks) = next(train_iter)
        try:
            (t_images, _, _) = next(tar_train_iter)
        except:
            tar_train_iter = iter(tar_dataloader)
            (t_images, _, _) = next(train_iter)

        # to device
        images = images.to(device)
        masks = masks.to(device)
        t_images = t_images.to(device)
        if num_iter % 400 == 0:
            visual = True
        else:
            visual = False
        # inference
        if num_iter > cfg['open_ins']:
            open_ins = True
            loss_dict, DA_LOSS = model(images, mask=masks, targets=targets, tar_x=t_images, open_ins=open_ins,
                                       visual=visual)
        else:
            open_ins = False
            loss_dict, DA_LOSS = model(images, mask=masks, targets=targets, tar_x=t_images, visual=visual)
        losses = loss_dict['losses']
        trainer_writer.add_scalar('train/losses', losses, ni)
        trainer_writer.add_scalar('train/loss_labels', loss_dict['loss_labels'], ni)
        trainer_writer.add_scalar('train/loss_bboxes', loss_dict['loss_bboxes'], ni)
        trainer_writer.add_scalar('train/loss_centerness', loss_dict['loss_centerness'], ni)
        trainer_writer.add_scalar('train/da_loss', loss_dict['da_loss'], ni)
        if open_ins:
            trainer_writer.add_scalar('train/ins_loss', loss_dict['ins_loss'], ni)
        # reduce
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
        if args.distributed:
            trainer_writer.add_scalar('train/losses_reduced', loss_dict_reduced['losses'], ni)
            trainer_writer.add_scalar('train/loss_labels_reduced', loss_dict_reduced['loss_labels'], ni)
            trainer_writer.add_scalar('train/loss_bboxes_reduced', loss_dict_reduced['loss_bboxes'], ni)
            trainer_writer.add_scalar('train/loss_centerness_reduced', loss_dict_reduced['loss_centerness'], ni)

        # check loss
        if torch.isnan(losses):
            print('loss is NAN !!')
            continue

        # Backward and Optimize
        losses.backward()
        if args.grad_clip_norm > 0.:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        else:
            total_norm = get_total_grad_norm(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

        # display
        if distributed_utils.is_main_process() and num_iter % 50 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
            cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1], 'lr_da': cur_lr[2]}
            log = dict(
                lr=round(cur_lr_dict['lr'], 6),
                lr_bk=round(cur_lr_dict['lr_bk'], 6)
            )

            s_h, s_w = images.shape[2:]
            t_h, t_w = t_images.shape[2:]
            # basic infor
            # log += '[Iter: {}]'.format(num_iter)
            # log += '[lr: {:.6f}][lr_bk: {:.6f}][lr_da: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'],
            #                                                            cur_lr_dict['lr_da'])
            # # loss infor
            # log += '[netD_loss: {:.3f}/{:.3f}/{:.3f}]'.format(DA_LOSS[0], DA_LOSS[1], DA_LOSS[2])
            # # other infor
            # log += '[time: {:.2f}]'.format(t1 - t0)
            # log += '[gnorm: {:.2f}]'.format(total_norm)
            # log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])
            # log += '[source_size: [{},{}]]'.format(s_h, s_w)
            # log += '[target_size: [{},{}]]'.format(t_h, t_w)
            logger.info(log)
            # print log infor
            print(log, flush=True)
            print(loss_dict,flush=True)
            t0 = time.time()

        lr_scheduler.step()
        if num_iter in cfg['epoch'][args.schedule]['lr_iter']:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * 0.1

        # evaluation
        if num_iter != 0:
            if (num_iter) % epoch_size == 0 or (num_iter) == max_epoch:
                # check evaluator
                if distributed_utils.is_main_process():
                    if evaluator is None:
                        print('No evaluator ... save model and go on training.')
                        print('Saving state, epoch: {}'.format(num_iter))
                        weight_name = '{}_epoch_{}.pth'.format(args.version, num_iter)
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save({'model': model_without_ddp.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': num_iter,
                                    'args': args},
                                   checkpoint_path)

                    else:
                        print('eval ...')
                        logger.info('eval ...')
                        # set eval mode
                        model_without_ddp.trainable = False
                        model_without_ddp.eval()

                        # evaluate
                        evaluator.evaluate(model_without_ddp, logger=logger)

                        cur_map = evaluator.map
                        map50 = evaluator.ap50
                        trainer_writer.add_scalar('train/map50', map50, num_iter)
                        if map50 > best_map:
                            # update best-map
                            best_map = map50
                            # save model
                            print('Saving state, epoch:', num_iter)
                            weight_name = '{}_epoch_{}_{:.2f}_ap50_{:.2f}.pth'.format(args.version, num_iter, cur_map * 100,
                                                                                      best_map * 100)
                            checkpoint_path = os.path.join(path_to_save, weight_name)
                            torch.save({'model': model_without_ddp.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'lr_scheduler': lr_scheduler.state_dict(),
                                        'epoch': num_iter,
                                        'args': args},
                                       checkpoint_path)

                            # set train mode.
                        model_without_ddp.trainable = True
                        model_without_ddp.train()

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

        # close mosaic augmentation
        if args.mosaic and max_epoch - num_iter == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False

    trainer_writer.close()


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      min_size=cfg['train_min_size'],
                                      max_size=cfg['train_max_size'],
                                      random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                      pixel_mean=cfg['pixel_mean'],
                                      pixel_std=cfg['pixel_std'],
                                      format=cfg['format'])
    target_transform = TrainTransforms(trans_config=trans_config,
                                       min_size=cfg['target_min_size'],
                                       max_size=cfg['target_max_size'],
                                       random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                       pixel_mean=cfg['pixel_mean'],
                                       pixel_std=cfg['pixel_std'],
                                       format=cfg['format'])
    val_transform = ValTransforms(min_size=cfg['test_min_size'],
                                  max_size=cfg['test_max_size'],
                                  pixel_mean=cfg['pixel_mean'],
                                  pixel_std=cfg['pixel_std'],
                                  format=cfg['format'],
                                  padding=cfg['val_padding'])
    color_augment = BaseTransforms(min_size=cfg['train_min_size'],
                                   max_size=cfg['train_max_size'],
                                   random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                   pixel_mean=cfg['pixel_mean'],
                                   pixel_std=cfg['pixel_std'],
                                   format=cfg['format'])
    # dataset
    global tar_dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        target_dir = os.path.join(args.root, 'foggy_cityscapes')
        num_classes = 20
        # dataset
        dataset = VOCDetection(img_size=cfg['train_max_size'],
                               data_dir=data_dir,
                               transform=train_transform,
                               color_augment=color_augment,
                               mosaic=args.mosaic)
        tar_dataset = VOCDetection(img_size=cfg['train_max_size'],
                                   data_dir=target_dir,
                                   transform=target_transform,
                                   color_augment=color_augment,
                                   mosaic=args.mosaic)
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        train_dir = os.path.join(args.root, 'kitti')
        target_dir = os.path.join(args.root, 'cityscapes')
        val_dir = os.path.join(args.root, 'cityscapes')
        num_classes = 1
        # dataset
        dataset = COCODataset(img_size=cfg['train_max_size'],
                              data_dir=train_dir,
                              image_set='kitti_train',
                              transform=train_transform,
                              color_augment=color_augment,
                              mosaic=args.mosaic)
        tar_dataset = COCODataset(img_size=cfg['train_max_size'],
                                  data_dir=target_dir,
                                  image_set='city_train',
                                  transform=target_transform,
                                  color_augment=color_augment,
                                  mosaic=args.mosaic)
        # evaluator
        evaluator = COCOAPIEvaluator(data_dir=val_dir,
                                     device=device,
                                     transform=val_transform)
        log = '[source_dataset: {}]'.format(train_dir)
        log += '[target_dataset: {}]'.format(target_dir)
        log += '[val_dataset: {}]'.format(val_dir)
        logger.info(log)

    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, tar_dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler,
                                                        args.batch_size,
                                                        drop_last=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn,
                                             num_workers=args.num_workers,
                                             )

    return dataloader


if __name__ == '__main__':
    train()
