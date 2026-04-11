import datetime
import gc
import operator
import os
import time
from functools import reduce

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn

from bert.modeling_bert import BertModel
from lib import segmentation

import metrics
import transforms as T
import utils


def get_dataset(image_set, transform, args):
    if args.dataset == 'plantseg':
        from data.dataset_plantseg import PlantSegDataset
        dataset = PlantSegDataset(args,
                                  split=image_set,
                                  image_transforms=transform,
                                  target_transforms=None,
                                  eval_mode=False)
    else:
        from data.dataset_refer_bert import ReferDataset
        dataset = ReferDataset(args,
                               split=image_set,
                               image_transforms=transform,
                               target_transforms=None)
    num_classes = 2
    return dataset, num_classes


def iou(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        score = 0
    else:
        score = float(intersection) / float(union)

    return score, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def criterion(input_tensor, target):
    weight = input_tensor.new_tensor([0.9, 1.1])
    return nn.functional.cross_entropy(input_tensor, target, weight=weight)


def forward_model(model, bert_model, image, sentences, attentions, return_aux=False):
    if bert_model is not None:
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
        embedding = last_hidden_states.permute(0, 2, 1)
        output = model(image, embedding, l_mask=attentions.unsqueeze(dim=-1), return_aux=return_aux)
        return output
    return model(image, sentences, l_mask=attentions, return_aux=return_aux)


def compute_hsb_loss(aux_dict, target):
    if not aux_dict:
        return torch.zeros((), device=target.device, dtype=torch.float32)

    hsb_loss = torch.zeros((), device=target.device, dtype=torch.float32)
    target = target.float().unsqueeze(1)
    for stage_name in ('hsb_stage3', 'hsb_stage4'):
        if stage_name not in aux_dict:
            continue
        m_h = aux_dict[stage_name]
        gt_i = F.interpolate(target, size=m_h.shape[-2:], mode='nearest')
        positive_mask = (gt_i > 0.5).float()
        target_zero = torch.zeros_like(m_h)
        per_pixel = F.binary_cross_entropy(m_h, target_zero, reduction='none')
        hsb_loss = hsb_loss + (per_pixel * positive_mask).sum() / (positive_mask.sum() + 1e-6)

    return hsb_loss


def evaluate(model, data_loader, bert_model, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    if args.dataset == 'plantseg':
        meter = metrics.BinarySegmentationMeter()
        with torch.no_grad():
            for data in metric_logger.log_every(data_loader, 100, header):
                image, target, sentences, attentions = data
                image = image.to(device, non_blocking=device.type == 'cuda')
                target = target.to(device, non_blocking=device.type == 'cuda')
                sentences = sentences.to(device, non_blocking=device.type == 'cuda').squeeze(1)
                attentions = attentions.to(device, non_blocking=device.type == 'cuda').squeeze(1)

                output = forward_model(model, bert_model, image, sentences, attentions)
                meter.update_from_logits(output, target)

        result = meter.compute()
        print('Final results:')
        print(metrics.format_binary_metrics(result))
        return result

    total_its = 0
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image = image.to(device, non_blocking=device.type == 'cuda')
            target = target.to(device, non_blocking=device.type == 'cuda')
            sentences = sentences.to(device, non_blocking=device.type == 'cuda').squeeze(1)
            attentions = attentions.to(device, non_blocking=device.type == 'cuda').squeeze(1)

            output = forward_model(model, bert_model, image, sentences, attentions)

            this_iou, intersection, union = iou(output, target)
            acc_ious += this_iou
            mean_IoU.append(this_iou)
            cum_I += intersection
            cum_U += union
            for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
            seg_total += 1
        avg_iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    overall_iou = float(cum_I * 100. / cum_U) if cum_U != 0 else 0.0
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
        results_str += '    precision@%s = %.2f\n' % (str(eval_seg_iou), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % overall_iou
    print(results_str)

    return 100 * avg_iou, overall_iou


def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, device, args):
    model.train()
    if bert_model is not None:
        bert_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, target, sentences, attentions = data
        image = image.to(device, non_blocking=device.type == 'cuda')
        target = target.to(device, non_blocking=device.type == 'cuda')
        sentences = sentences.to(device, non_blocking=device.type == 'cuda').squeeze(1)
        attentions = attentions.to(device, non_blocking=device.type == 'cuda').squeeze(1)

        if args.use_hsb:
            output, aux_dict = forward_model(model, bert_model, image, sentences, attentions, return_aux=True)
            seg_loss = criterion_fn(output, target)
            hsb_loss = compute_hsb_loss(aux_dict, target)
            loss = seg_loss + args.lambda_hsb * hsb_loss
        else:
            output = forward_model(model, bert_model, image, sentences, attentions)
            seg_loss = criterion_fn(output, target)
            hsb_loss = None
            loss = seg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        iterations += 1
        if args.use_hsb:
            metric_logger.update(loss=loss.item(),
                                 seg_loss=seg_loss.item(),
                                 hsb_loss=hsb_loss.item(),
                                 lr=optimizer.param_groups[0]["lr"])
        else:
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, seg_loss, output, data
        if args.use_hsb:
            del aux_dict, hsb_loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset, _ = get_dataset("train", get_transform(args=args), args=args)
    dataset_val, _ = get_dataset("val", get_transform(args=args), args=args)

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers,
        pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=val_sampler, num_workers=args.workers)

    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model = model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
    single_model = model.module if args.distributed else model

    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        bert_model = bert_model.to(device)
        if args.distributed:
            bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
            bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module if args.distributed else bert_model
    else:
        bert_model = None
        single_bert_model = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    backbone_no_decay = []
    backbone_decay = []
    for name, parameter in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1 - step / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    iterations = 0
    best_score = -1.0

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch,
                        args.print_freq, iterations, bert_model, device, args)
        validation_result = evaluate(model, data_loader_val, bert_model, device, args)

        if args.dataset == 'plantseg':
            print('Foreground IoU {}'.format(validation_result['fg_iou'] * 100.0))
            print('mIoU {}'.format(validation_result['miou'] * 100.0))
            current_score = validation_result['fg_iou']
        else:
            avg_iou, overall_iou = validation_result
            print('Average object IoU {}'.format(avg_iou))
            print('Overall IoU {}'.format(overall_iou))
            current_score = overall_iou / 100.0

        if best_score < current_score:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {
                    'model': single_model.state_dict(),
                    'bert_model': single_bert_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'lr_scheduler': lr_scheduler.state_dict()
                }
            else:
                dict_to_save = {
                    'model': single_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'lr_scheduler': lr_scheduler.state_dict()
                }

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_score = current_score

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
