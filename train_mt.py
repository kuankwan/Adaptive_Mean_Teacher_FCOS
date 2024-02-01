from __future__ import division

from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
import random
from copy import deepcopy
import math
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.transforms import TrainTransforms, ValTransforms, BaseTransforms, build_strong_augmentation, \
    StrongTransforms
from models.backbone.resnet import Backbone
from models.backbone.vision.model_resnet import ResNet

from models.neck import fpn
from utils.misc import cos_loss, QFLLoss, sigmoid_varifocal_loss, permute_to_N_HWA_K, QFLv2, dice_loss, infoNCE_loss, \
    sigmoid_focal_loss, kl_divergence
from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler
from utils.solver.warmup_schedule import build_warmup
from mt import *
from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from models.detector.fcos.fcos_mt import FCOS_MT
from config import build_config
from models.detector import build_model
from utils.logger import *
from tqdm import tqdm
from datetime import datetime
import time
from utils.iou_loss import iou_loss

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
    parser.add_argument('-rtea', '--resumetea', default=None, type=str,
                        help='keep teacher training')

    # dataset
    parser.add_argument('--root', default='D:\\datasets',
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
    parser.add_argument("--local_rank", type=int, help="")

    return parser.parse_args()


def loss_unsupvise(stu,tea,beta=2.0):

    loss = F.binary_cross_entropy_with_logits(stu, tea, reduction='none')
    preds_sigmoid = stu.sigmoid()
    scale_factor = (preds_sigmoid - tea).abs().pow(beta)
    loss = loss * scale_factor
    loss = loss.mean()
    return loss

@torch.no_grad()
def _copy_main_model(model_teacher, model):
    # initialize all parameters
    if get_world_size() > 1:
        rename_model_dict = {
            key[7:]: value for key, value in model.state_dict().items()
        }
        new_model = OrderedDict()
        for key, value in rename_model_dict.items():
            if key in model_teacher.keys():
                new_model[key] = rename_model_dict[key]
        model_teacher.load_state_dict(new_model)
        # model_teacher.load_state_dict(rename_model_dict)
    else:
        new_model = OrderedDict()
        for key, value in model.state_dict().items():
            if key in model_teacher.state_dict().keys():
                new_model[key] = value
        model_teacher.load_state_dict(new_model)


@torch.no_grad()
def _update_teacher_model(model, model_teacher, keep_rate=0.9):
    if get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model.state_dict().items()
        }
    else:
        student_model_dict = model.state_dict()
    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        # else:
        #     raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)


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
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # config
    cfg = build_config(args)
    for key, value in cfg.items():
        log_info = f'{key}: {value}'
        logger.info(log_info)

    # dataset and evaluator
    dataset, tar_dataset, evaluator, num_classes = build_dataset(cfg, args, device)
    evaluator_t = evaluator

    # dataloader
    # dataloader = build_dataloader(args, dataset, CollateFunc())
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

    iterative_steps = 5  # 迭代式剪枝，重复5次Pruning-Finetuning的循环完成剪枝。
    ignored_layers = []
    for m in model.modules():
        if  m is model.cls_pred or m is model.reg_pred or m is model.ctn_pred:
            ignored_layers.append(m)  # DO NOT prune the final classifier!
    print(ignored_layers)

    if 'fcos' in args.version:
        teacher_model = FCOS_MT(cfg=cfg,
                     device=device,
                     num_classes=num_classes,
                     trainable=True,
                     conf_thresh=cfg['conf_thresh'],
                     nms_thresh=cfg['train_nms_thresh'],
                     topk=args.topk)
        teacher_model = teacher_model.to(device).train()
        for name, param in teacher_model.named_parameters():
            log_info = f'name: {name}, grad: {param.requires_grad}'
            logger.info(log_info)
        if args.resumetea is not None:
            print('keep training: ', args.resumetea)
            tea_checkpoint = torch.load(args.resumetea, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = tea_checkpoint.pop("model")
            teacher_model.load_state_dict(checkpoint_state_dict)
    else:
        teacher_model = None
    model = model.to(device).train()
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info)



    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)

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

    # ATeacher framwork manager
    ATMGR = ATeacherTrainer(keep_rate=0.996)

    # tensorboard
    path_to_write = os.path.join(path_to_save, 'log_loss')
    os.makedirs(path_to_write, exist_ok=True)
    trainer_writer = SummaryWriter(log_dir=path_to_write)

    # training configuration
    max_epoch = cfg['epoch'][args.schedule]['max_epoch']
    epoch_size = len(dataloader)
    print("target data size:%d" % (len(tar_dataloader)))
    best_map = -1.
    best_map_t = -1.
    best_map_all = -1.
    best_map_all_t = -1.
    # print("best_map:{}".format(best_map))
    warmup = not args.no_warmup
    t0 = time.time()
    total_epoch = epoch_size * max_epoch
    # start training loop
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)
            tar_dataloader.batch_sampler.sampler.set_epoch(epoch)
        bar = tqdm(range(epoch_size))
        train_iter = iter(dataloader)
        tar_train_iter = iter(tar_dataloader)
        visual = False



        # train one epoch
        # for iter_i, (images, targets, masks) in enumerate(dataloader):
        for step, num_iter in enumerate(bar):
            ni = num_iter + (epoch - 1) * epoch_size

            if ni < cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, args.base_lr, args.base_lr)
            # data
            try:
                (images,_, targets, masks) = next(train_iter)
            except:
                train_iter = iter(dataloader)
                (images,_, targets, masks) = next(train_iter)
            try:
                (t_images,t_weak_images, _, _) = next(tar_train_iter)
            except:
                tar_train_iter = iter(tar_dataloader)
                (t_images,t_weak_images, _, _) = next(tar_train_iter)


            # to device
            images = images.to(device)
            masks = masks.to(device)
            t_images = t_images.to(device)
            if t_weak_images is not None:
                w_images = t_weak_images.to(device)
            # t_images = None
            if num_iter % 1000 == 0:
                visual = True
            else:
                visual = False
            visual = False
            # inference
            if epoch >= cfg['open_ins']:
                loss_dict,_ = model(images, mask=masks, targets=targets, tar_x=t_images, open_ins=True, visual=False)
            else:
                loss_dict,_ = model(images, mask=masks, targets=targets, tar_x=t_images, visual=False)


            pga_weight = 2 / (1 + math.exp(- cfg['ga.mmaa.weight'] * ni / cfg['teacher_training'])) * cfg['daloss_alpha']
            pga_loss = loss_dict['pga_loss']  * pga_weight
            losses = loss_dict['losses'] + pga_loss
            loss_dict.update({'losses': losses, 'pga_loss': pga_loss})



            # reduce
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            for k,v in loss_dict_reduced.items():
                trainer_writer.add_scalar(k, v, ni)
            # trainer_writer.add_scalar('train/loss_labels_reduced', loss_dict_reduced['loss_labels'], ni)
            # trainer_writer.add_scalar('train/loss_bboxes_reduced', loss_dict_reduced['loss_bboxes'], ni)
            # trainer_writer.add_scalar('train/loss_centerness_reduced', loss_dict_reduced['loss_centerness'], ni)
            if args.distributed:
                trainer_writer.add_scalar('train/losses_reduced', loss_dict_reduced['losses'], ni)
                trainer_writer.add_scalar('train/loss_labels_reduced', loss_dict_reduced['loss_labels'], ni)
                trainer_writer.add_scalar('train/loss_bboxes_reduced', loss_dict_reduced['loss_bboxes'], ni)
                trainer_writer.add_scalar('train/loss_centerness_reduced', loss_dict_reduced['loss_centerness'], ni)
                # trainer_writer.add_scalar('train/loss_PGA', loss_dict_reduced['pga_loss'], ni)
                # trainer_writer.add_scalar('train/loss_CLAA', loss_dict_reduced['local_loss'], ni)
                # trainer_writer.add_scalar('train/loss_CGAA', loss_dict_reduced['global_loss'], ni)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue
            # Backward and Optimize
            losses.backward()
            losses = losses.detach()




            if ni >= cfg['teacher_training'] and teacher_model is not None and w_images is not None:
                uns_loss = torch.tensor(0, dtype=torch.float).to(device)
                _, stu_target_outputs = model(images, mask=masks, targets=targets, tar_x=t_images, visual=visual,weak_train=True)
                _, stu_source_outputs = model(images, mask=masks, targets=targets, tar_x=None, visual=False,
                                       weak_train=False)
                with torch.no_grad():
                    tea_outputs = teacher_model(w_images, mask=None, visual=False)
                tea_pred_cls, tea_pred_delta, tea_pred_ctn = tea_outputs['pred_cls'],tea_outputs['pred_reg'],tea_outputs['pred_ctn']
                tea_pred_cls = torch.cat([permute_to_N_HWA_K(tea_cls,num_classes) for tea_cls in tea_pred_cls],dim=1).view(-1,num_classes)
                tea_pred_delta = torch.cat([permute_to_N_HWA_K(tea_delta, 4) for tea_delta in tea_pred_delta],
                                         dim=1).view(-1, 4)
                tea_pred_ctn = torch.cat([permute_to_N_HWA_K(tea_ctn, 1) for tea_ctn in tea_pred_ctn],
                                         dim=1).view(-1, 1)
                stu_target_pred_cls = torch.cat([permute_to_N_HWA_K(stu_cls, num_classes) for stu_cls in stu_target_outputs['pred_cls']],
                                         dim=1).view(-1, num_classes)
                stu_source_pred_cls = torch.cat(
                    [permute_to_N_HWA_K(stu_cls, num_classes) for stu_cls in stu_source_outputs['pred_cls']],
                    dim=1).view(-1, num_classes)
                stu_source_pred_ctn = torch.cat(
                    [permute_to_N_HWA_K(stu_cls, 1) for stu_cls in stu_source_outputs['pred_ctn']],
                    dim=1).view(-1, 1)
                stu_target_pred_deltas = torch.cat([permute_to_N_HWA_K(stu_delta, 4) for stu_delta in stu_target_outputs['pred_reg']],
                                           dim=1).view(-1, 4)
                stu_target_pred_ctn = torch.cat([permute_to_N_HWA_K(stu_ctn, 1) for stu_ctn in stu_target_outputs['pred_ctn']],
                                         dim=1).view(-1, 1)
                stu_target_mae_mask = torch.cat(
                    [permute_to_N_HWA_K(stu_ctn, 1) for stu_ctn in stu_target_outputs['mae_target']],
                    dim=1).view(-1, 1)
                stu_source_mae_mask = torch.cat(
                    [permute_to_N_HWA_K(stu_ctn, 1) for stu_ctn in stu_target_outputs['mae_source']],
                    dim=1).view(-1, 1)
                stu_pred_cr = torch.cat(
                    [permute_to_N_HWA_K(cr, num_classes) for cr in stu_target_outputs['pred_cr']],
                    dim=1).view(-1, num_classes)
                stu_source_mask = torch.cat(stu_source_outputs['mask'],dim=1).view(-1,1)
                with torch.no_grad():
                    ratio = 0.01
                    count_num = int(tea_pred_cls.size(0) * ratio)
                    teacher_probs = tea_pred_cls.sigmoid() * tea_pred_ctn.sigmoid()
                    max_vals = torch.max(teacher_probs, 1)[0]    # size[30804]
                    sorted_vals, sorted_inds = torch.topk(max_vals, tea_pred_cls.size(0))
                    mask = torch.zeros_like(max_vals)
                    mask[sorted_inds[:count_num]] = 1.
                    num_foreground = mask.sum()
                    fg_num = sorted_vals[:count_num].sum()
                    b_mask = mask > 0.

                loss_logits = QFLv2(
                    stu_target_pred_cls.sigmoid(),
                    tea_pred_cls.sigmoid(),
                    weight=mask,
                    reduction="sum",
                ) / fg_num

                loss_dice = dice_loss(stu_target_pred_cls.sigmoid(), mask,)
                loss_dis_kl = torch.mean(torch.abs(F.kl_div(stu_source_pred_cls.sigmoid(),stu_pred_cr.sigmoid(),reduction='none') -
                                         F.kl_div(stu_pred_cr.sigmoid(),stu_target_pred_cls.sigmoid(),reduction='none')))
                loss_deltas = iou_loss(
                    stu_target_pred_deltas[b_mask],
                    tea_pred_delta[b_mask],
                    box_mode="ltrb",
                    loss_type='giou',
                    reduction="mean",
                )


                loss_quality = F.binary_cross_entropy(
                    stu_target_pred_ctn[b_mask].sigmoid(),
                    tea_pred_ctn[b_mask].sigmoid(),
                    reduction='sum'
                ) / num_foreground


                mask_target_label = torch.full(stu_target_mae_mask.shape, 0.0, dtype=torch.float, device=device)
                mask_source_label = torch.full(stu_source_mae_mask.shape, 1.0, dtype=torch.float, device=device)

                loss_mae_mask = sigmoid_focal_loss(stu_source_mae_mask, mask_source_label,alpha=-1,
                                                                   reduction="mean") + \
                                sigmoid_focal_loss(stu_target_mae_mask, mask_target_label,alpha=-1,
                                                                   reduction="mean")
                mae_mask_weight = pga_weight - 1

                unsloss_dict = {
                    "uns_loss_logits": loss_logits,
                    "uns_loss_deltas": loss_deltas,
                    "uns_loss_quality": loss_quality,
                    "uns_loss_dice": loss_dice,
                    "loss_mae_mask": loss_mae_mask,
                    "loss_kl_dis":loss_dis_kl,
                }
                distill_weights = {
                    "uns_loss_logits": cfg['DISTILL.WEIGHTS.LOGITS'],
                    "uns_loss_deltas": cfg['DISTILL.WEIGHTS.DELTAS'],
                    "uns_loss_quality": cfg['DISTILL.WEIGHTS.QUALITY'],
                    "uns_loss_dice": cfg['DISTILL.WEIGHTS.DICE'],
                    "loss_mae_mask": cfg['DISTILL.WEIGHTS.MAEMASK'] * mae_mask_weight,
                    "loss_kl_dis": cfg['DISTILL.WEIGHTS.KL'],
                    "fore_ground_sum": 1.,
                }
                loss_dict_unsup = {k: v * distill_weights[k] for k, v in unsloss_dict.items()}
                uns_losses = sum([
                    metrics_value for metrics_value in loss_dict_unsup.values()
                    if metrics_value.requires_grad
                ])
                loss_dict_unsup__reduced = distributed_utils.reduce_dict(loss_dict_unsup)
                loss_dict_reduced.update(loss_dict_unsup__reduced)
                trainer_writer.add_scalar('train/uns_losses', uns_losses, ni)
                for k,v in loss_dict_unsup__reduced.items():
                    trainer_writer.add_scalar(k,v,ni)
                # Backward and Optimize
                uns_losses.backward()
                loss_dict.update(loss_dict_unsup)
                losses += uns_losses.detach()
                loss_dict.update({'losses': losses,'uns_losses':uns_losses})




            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            #EMA
            if ni == cfg['teacher_training']:
                teacher_model, _ = ATMGR._copy_main_model(model, teacher_model)
                print("===================================")
                print("<- init teacher model ->")
                print("===================================")
            elif ni > cfg['teacher_training']:
                teacher_model, _ = ATMGR._update_teacher_model(model, teacher_model)

            # if epoch==2:
            #     pruner = tp.pruner.MetaPruner(
            #         model,
            #         torch.randn((2,3,608,1216)),  # 用于分析依赖的伪输入
            #         importance=imp,  # 重要性评估指标
            #         iterative_steps=iterative_steps,  # 迭代剪枝，设为1则一次性完成剪枝
            #         pruning_ratio=0.5,
            #         # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            #         ignored_layers=ignored_layers,  # 忽略掉最后的分类层
            #     )
            #     base_macs, base_nparams = tp.utils.count_ops_and_params(model, images)
            #     for i in range(iterative_steps):
            #         pruner.step()  # 执行裁剪，本例子中我们每次会裁剪10%，共执行5次，最终稀疏度为50%
            #         macs, nparams = tp.utils.count_ops_and_params(model, images)
            #         print("  Iter %d/%d, Params: %.2f M => %.2f M" % (
            #         i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
            #         print(
            #             "  Iter %d/%d, MACs: %.2f G => %.2f G" % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9))




                # display
            if distributed_utils.is_main_process() and num_iter % 50 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                log = dict(
                    lr=round(cur_lr_dict['lr'], 6),
                    lr_bk=round(cur_lr_dict['lr_bk'], 6)
                )

                s_h, s_w = images.shape[2:]
                # basic infor
                log = '[Epoch: {}/{}]'.format(epoch, max_epoch - 1)
                log += '[Iter: {}/{}]'.format(num_iter, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
                # loss infor
                for k in loss_dict.keys():
                    log += '[{}: {:.3f}]'.format(k, loss_dict[k])
                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[gnorm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])
                log += '[source_size: [{},{}]]'.format(s_h, s_w)
                logger.info(log)
                # print log infor
                print(log, flush=True)

                t0 = time.time()

        lr_scheduler.step()
        if epoch in cfg['epoch'][args.schedule]['lr_epoch']:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * 0.1

        # evaluation
        if (epoch) % args.eval_epoch == 0 or (epoch) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
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

                    trainer_writer.add_scalar('train/map50', map50, epoch)
                    if map50 >= best_map or cur_map >= best_map_all:
                        # update best-map
                        best_map = map50
                        best_map_all = cur_map
                        # save model
                        print('Saving state, epoch:', epoch)
                        weight_name = '{}_epoch_{}_{:.2f}_ap50_{:.2f}.pth'.format(args.version, epoch, cur_map * 100,
                                                                                  best_map * 100)
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save({'model': model_without_ddp.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch,
                                    'args': args},
                                   checkpoint_path)

                    teacher_model.trainable = False
                    teacher_model.eval()
                    with torch.no_grad():
                        evaluator.evaluate(teacher_model, logger=logger)
                    t_cur_map = evaluator.map
                    t_map50 = evaluator.ap50
                    if t_map50 >= best_map or t_cur_map >= best_map_all:
                        # update best-map
                        best_map_t = t_map50
                        best_map_all_t = t_cur_map
                        # save model
                        print('Saving state, epoch:', epoch)
                        weight_name = 'T_{}_epoch_{}_{:.2f}_ap50_{:.2f}.pth'.format(args.version, epoch,
                                                                                  best_map_all_t * 100,
                                                                                  best_map_t * 100)
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save({'model': teacher_model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch,
                                    'args': args},
                                   checkpoint_path)

                        # set train mode.
                    model_without_ddp.trainable = True
                    model_without_ddp.train()
                    teacher_model.trainable = True
                    teacher_model.train()

            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

        # close mosaic augmentation
        if args.mosaic and max_epoch - epoch == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False

    trainer_writer.close()


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = StrongTransforms(trans_config=trans_config,
                                      min_size=cfg['train_min_size'],
                                      max_size=cfg['train_max_size'],
                                      random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                      pixel_mean=cfg['pixel_mean'],
                                      pixel_std=cfg['pixel_std'],
                                      format=cfg['format'])
    target_transform = StrongTransforms(trans_config=trans_config,
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
    weak_augment = TrainTransforms(trans_config=trans_config,
                                   min_size=cfg['train_min_size'],
                                   max_size=cfg['train_max_size'],
                                   random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                   pixel_mean=cfg['pixel_mean'],
                                   pixel_std=cfg['pixel_std'],
                                   format=cfg['format'])
    # dataset
    global tar_dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOC2012_1')
        target_dir = os.path.join(args.root, 'comic')
        val_dir = os.path.join(args.root, 'comic')
        num_classes = 6
        # dataset
        dataset = VOCDetection(img_size=cfg['train_max_size'],
                               data_dir=data_dir,
                               transform=train_transform,
                               color_augment=color_augment,
                               strong=build_strong_augmentation(),
                               mosaic=args.mosaic)
        tar_dataset = VOCDetection(img_size=cfg['train_max_size'],
                                   data_dir=target_dir,
                                   transform=target_transform,
                                   color_augment=color_augment,
                                   strong=build_strong_augmentation(),
                                   weak=weak_augment,
                                   mosaic=args.mosaic)
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=val_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        train_dir = os.path.join(args.root, 'cityscapes')
        target_dir = os.path.join(args.root, 'foggy_cityscapes')
        val_dir = os.path.join(args.root, 'foggy_cityscapes')
        num_classes = 8
        # dataset
        dataset = COCODataset(img_size=cfg['train_max_size'],
                              data_dir=train_dir,
                              image_set='train',
                              transform=train_transform,
                              color_augment=color_augment,
                              mosaic=args.mosaic,
                              strong=build_strong_augmentation())
        tar_dataset = COCODataset(img_size=cfg['train_max_size'],
                                  data_dir=target_dir,
                                  image_set='tar_train',
                                  transform=target_transform,
                                  color_augment=color_augment,
                                  mosaic=args.mosaic,
                                  weak=weak_augment,
                                  strong=build_strong_augmentation())

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
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn,
                                             num_workers=args.num_workers)

    return dataloader


if __name__ == '__main__':
    train()
