import datetime
import gc
import operator
import os
import time
from functools import reduce

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
from lib import segmentation

import transforms as T
import utils


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset

    dataset = ReferDataset(
        args,
        split=image_set,
        image_transforms=transform,
        target_transforms=None,
    )
    num_classes = 2
    return dataset, num_classes


def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(transforms)


def build_boundary_target(target):
    target = target.float().unsqueeze(1)
    dilated = nn.functional.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - nn.functional.max_pool2d(1.0 - target, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp_(0.0, 1.0)


def criterion(input_tensor, target, boundary_logits=None, boundary_loss_weight=0.0):
    weight = input_tensor.new_tensor([0.9, 1.1])
    seg_loss = nn.functional.cross_entropy(input_tensor, target, weight=weight)
    loss_dict = {
        'loss': seg_loss.detach().item(),
        'seg_loss': seg_loss.detach().item(),
    }

    if boundary_logits is None or boundary_loss_weight <= 0.0:
        return seg_loss, loss_dict

    if boundary_logits.shape[-2:] != target.shape[-2:]:
        boundary_logits = nn.functional.interpolate(
            boundary_logits,
            size=target.shape[-2:],
            mode='bilinear',
            align_corners=True,
        )
    boundary_target = build_boundary_target(target)
    boundary_loss = nn.functional.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
    total_loss = seg_loss + boundary_loss_weight * boundary_loss
    loss_dict['boundary_loss'] = boundary_loss.detach().item()
    loss_dict['loss'] = total_loss.detach().item()
    return total_loss, loss_dict


def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, target, sentences, attentions = data
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sentences = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            attentions = attentions.unsqueeze(dim=-1)
            output = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)

        classifier = model.module.classifier if hasattr(model, 'module') else model.classifier
        boundary_logits = getattr(classifier, 'last_boundary_logits', None)
        loss, loss_dict = criterion_fn(
            output,
            target,
            boundary_logits=boundary_logits,
            boundary_loss_weight=args.boundary_loss_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        iterations += 1
        metric_logger.update(loss=loss_dict['loss'], seg_loss=loss_dict['seg_loss'],
                             lr=optimizer.param_groups[0]['lr'])
        if 'boundary_loss' in loss_dict:
            metric_logger.update(boundary_loss=loss_dict['boundary_loss'])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return iterations


def build_checkpoint(args, epoch, single_model, optimizer, lr_scheduler, single_bert_model=None):
    checkpoint = {
        'model': single_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    if single_bert_model is not None:
        checkpoint['bert_model'] = single_bert_model.state_dict()
    return checkpoint


def sync_decoder_args_from_checkpoint(args, checkpoint_args):
    if checkpoint_args is None:
        return
    for name in ['decoder_head', 'use_boundary_refine', 'boundary_loss_weight', 'boundary_alpha']:
        if hasattr(checkpoint_args, name):
            current_value = getattr(args, name, None)
            default_value = {
                'decoder_head': 'simple',
                'use_boundary_refine': False,
                'boundary_loss_weight': 0.0,
                'boundary_alpha': 0.1,
            }[name]
            if current_value == default_value:
                setattr(args, name, getattr(checkpoint_args, name))


def main(args):
    dataset, _ = get_dataset('train', get_transform(args=args), args=args)

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        train_sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        sync_decoder_args_from_checkpoint(args, resume_checkpoint.get('args'))

    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True,
        )
        single_model = model.module
    else:
        single_model = model

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        bert_model.cuda()
        if args.distributed:
            bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
            bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
            single_bert_model = bert_model.module
        else:
            single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None

    if resume_checkpoint is not None:
        single_model.load_state_dict(resume_checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(resume_checkpoint['bert_model'])

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
            {'params': [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {'params': reduce(
                operator.concat,
                [[p for p in single_bert_model.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)],
            )},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {'params': [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {'params': reduce(
                operator.concat,
                [[p for p in single_model.text_encoder.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)],
            )},
        ]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (1 - step / (len(data_loader) * args.epochs)) ** 0.9,
    )

    start_time = time.time()
    iterations = 0

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler'])
        resume_epoch = resume_checkpoint['epoch']
    else:
        resume_epoch = -1

    print('Validation disabled: training uses train.json only. Run test.py separately for test.json evaluation.')

    last_checkpoint = None
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iterations = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            lr_scheduler,
            epoch,
            args.print_freq,
            iterations,
            bert_model,
            args,
        )
        last_checkpoint = build_checkpoint(
            args,
            epoch,
            single_model,
            optimizer,
            lr_scheduler,
            single_bert_model=single_bert_model,
        )
        utils.save_on_master(
            last_checkpoint,
            os.path.join(args.output_dir, 'checkpoint_last_{}.pth'.format(args.model_id)),
        )

    if last_checkpoint is None:
        last_checkpoint = build_checkpoint(
            args,
            resume_epoch,
            single_model,
            optimizer,
            lr_scheduler,
            single_bert_model=single_bert_model,
        )

    utils.save_on_master(
        last_checkpoint,
        os.path.join(args.output_dir, 'model_final_{}.pth'.format(args.model_id)),
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
