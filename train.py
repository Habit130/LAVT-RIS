import datetime
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from eval_ris_metrics import evaluate_mask_arrays, format_metrics_summary
from lib import segmentation
from text_encoder import (TEXT_ENCODER_MODEL_KEY, build_text_encoder, encode_text,
                          get_checkpoint_text_encoder_state, prepare_text_encoder_args)
import transforms as T
import utils


CHECKPOINT_CONFIG_KEY = 'checkpoint_config'
CHECKPOINT_CONFIG_FIELDS = (
    'ablation_config',
    'align_module',
    'gate_module',
    'hlg_stages',
    'swin_type',
    'model',
    'dataset',
    'text_encoder_name',
    'max_text_tokens',
    'plantseg_caption_index',
    'img_size',
)


def collect_checkpoint_config(args):
    config = {}
    for field in CHECKPOINT_CONFIG_FIELDS:
        value = getattr(args, field)
        if field == 'hlg_stages':
            value = [int(item) for item in value]
        config[field] = value
    return config


def validate_checkpoint_config(checkpoint, args, checkpoint_path, context):
    current_config = collect_checkpoint_config(args)
    print('{} checkpoint path: {}'.format(context, checkpoint_path))
    print('{} strict load enabled: True'.format(context))
    print('{} current args config: {}'.format(context, json.dumps(current_config, sort_keys=True)))

    if CHECKPOINT_CONFIG_KEY not in checkpoint:
        raise ValueError(
            '{} checkpoint [{}] is missing [{}]; it cannot be used for strict {}. '
            'Use a new-format checkpoint or convert it after manually confirming the config.'
            .format(context, checkpoint_path, CHECKPOINT_CONFIG_KEY, context.lower())
        )

    loaded_config = checkpoint[CHECKPOINT_CONFIG_KEY]
    print('{} loaded checkpoint config: {}'.format(context, json.dumps(loaded_config, sort_keys=True)))

    mismatches = []
    for field in CHECKPOINT_CONFIG_FIELDS:
        loaded_value = loaded_config.get(field)
        current_value = current_config[field]
        if loaded_value != current_value:
            mismatches.append((field, loaded_value, current_value))

    if mismatches:
        details = '; '.join(
            '{}: checkpoint={!r}, current={!r}'.format(field, loaded_value, current_value)
            for field, loaded_value, current_value in mismatches
        )
        print('{} config match result: mismatch'.format(context))
        raise ValueError('{} checkpoint config mismatch: {}'.format(context, details))

    print('{} config match result: match'.format(context))


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


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def criterion(input_tensor, target):
    weight = input_tensor.new_tensor([0.9, 1.1])
    return nn.functional.cross_entropy(input_tensor, target, weight=weight)


def unpack_batch(data):
    if len(data) == 6:
        image, target, sentences, attentions, false_healthy_mask, has_false_healthy = data
    else:
        image, target, sentences, attentions = data
        false_healthy_mask = None
        has_false_healthy = None
    return image, target, sentences, attentions, false_healthy_mask, has_false_healthy


def compute_gate_losses(aux_outputs, disease_mask, false_healthy_mask, has_false_healthy, false_healthy_weight):
    zero = torch.zeros((), device=disease_mask.device, dtype=torch.float32)
    if not aux_outputs or false_healthy_mask is None or has_false_healthy is None:
        return {
            'gate_loss': zero,
            'gate_disease_loss': zero,
            'gate_false_healthy_loss': zero,
            'gate_stage3_mean': zero,
            'gate_stage4_mean': zero,
        }

    disease_mask = disease_mask.unsqueeze(1).float()
    false_healthy_mask = false_healthy_mask.unsqueeze(1).float()
    sample_valid = has_false_healthy.float().view(-1, 1, 1, 1)

    gate_loss = zero
    gate_disease_loss = zero
    gate_false_healthy_loss = zero
    stage_means = {
        'gate_stage3_mean': zero,
        'gate_stage4_mean': zero,
    }

    for stage_name in ('hlg_stage3', 'hlg_stage4'):
        gate_tensor = aux_outputs.get(stage_name)
        if gate_tensor is None:
            continue

        gate_map = gate_tensor.mean(dim=1, keepdim=True)
        disease_i = F.interpolate(disease_mask, size=gate_map.shape[-2:], mode='nearest')
        false_healthy_i = F.interpolate(false_healthy_mask, size=gate_map.shape[-2:], mode='nearest')

        disease_pos = (disease_i > 0.5).float()
        false_pos = (false_healthy_i > 0.5).float()

        pos_target = torch.ones_like(gate_map)
        per_pixel_dis = (gate_map - pos_target) ** 2
        loss_dis_i = (per_pixel_dis * disease_pos).sum() / (disease_pos.sum() + 1e-6)

        false_healthy_target = torch.zeros_like(gate_map)
        per_pixel_fh = (gate_map - false_healthy_target) ** 2
        valid_false_pos = false_pos * sample_valid
        loss_fh_i = (per_pixel_fh * valid_false_pos).sum() / (valid_false_pos.sum() + 1e-6)

        gate_loss = gate_loss + loss_dis_i + false_healthy_weight * loss_fh_i
        gate_disease_loss = gate_disease_loss + loss_dis_i
        gate_false_healthy_loss = gate_false_healthy_loss + loss_fh_i
        stage_means[f'gate_{stage_name.split("_")[-1]}_mean'] = gate_map.mean().detach()

    return {
        'gate_loss': gate_loss,
        'gate_disease_loss': gate_disease_loss,
        'gate_false_healthy_loss': gate_false_healthy_loss,
        'gate_stage3_mean': stage_means['gate_stage3_mean'],
        'gate_stage4_mean': stage_means['gate_stage4_mean'],
    }


def should_return_aux(args):
    return getattr(args, 'gate_module', 'none') == 'hlg'


def forward_model(model, text_encoder, image, sentences, attentions, return_aux=False):
    if text_encoder is not None:
        last_hidden_states = encode_text(text_encoder, sentences, attentions)
        embedding = last_hidden_states.permute(0, 2, 1)
        output = model(image, embedding, l_mask=attentions.unsqueeze(dim=-1), return_aux=return_aux)
        return output
    return model(image, sentences, l_mask=attentions, return_aux=return_aux)


def evaluate(model, data_loader, text_encoder, device, args):
    model.eval()
    if text_encoder is not None:
        text_encoder.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    pred_masks = []
    gt_masks = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, _, _ = unpack_batch(data)
            image = image.to(device, non_blocking=device.type == 'cuda')
            target = target.to(device, non_blocking=device.type == 'cuda')
            sentences = sentences.to(device, non_blocking=device.type == 'cuda').squeeze(1)
            attentions = attentions.to(device, non_blocking=device.type == 'cuda').squeeze(1)

            output = forward_model(model, text_encoder, image, sentences, attentions)
            prediction = output.argmax(1).detach().cpu().tolist()
            target_list = target.detach().cpu().tolist()
            pred_masks.extend(prediction)
            gt_masks.extend(target_list)

    result = evaluate_mask_arrays(pred_masks, gt_masks)
    print('Final results:')
    print(format_metrics_summary(result))
    return result


def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, text_encoder, device, args):
    model.train()
    if text_encoder is not None:
        text_encoder.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, target, sentences, attentions, false_healthy_mask, has_false_healthy = unpack_batch(data)
        image = image.to(device, non_blocking=device.type == 'cuda')
        target = target.to(device, non_blocking=device.type == 'cuda')
        sentences = sentences.to(device, non_blocking=device.type == 'cuda').squeeze(1)
        attentions = attentions.to(device, non_blocking=device.type == 'cuda').squeeze(1)
        if false_healthy_mask is not None:
            false_healthy_mask = false_healthy_mask.to(device, non_blocking=device.type == 'cuda')
            has_false_healthy = has_false_healthy.to(device, non_blocking=device.type == 'cuda')

        if should_return_aux(args):
            output, aux_outputs = forward_model(model, text_encoder, image, sentences, attentions, return_aux=True)
        else:
            output = forward_model(model, text_encoder, image, sentences, attentions)
            aux_outputs = {}
        seg_loss = criterion_fn(output, target)
        gate_losses = compute_gate_losses(aux_outputs, target, false_healthy_mask, has_false_healthy,
                                          args.hlg_false_healthy_weight)
        loss = seg_loss + args.hlg_aux_loss_weight * gate_losses['gate_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        iterations += 1
        metric_logger.update(loss=loss.item(),
                             seg_loss=seg_loss.item(),
                             gate_loss=gate_losses['gate_loss'].item(),
                             gate_disease_loss=gate_losses['gate_disease_loss'].item(),
                             gate_false_healthy_loss=gate_losses['gate_false_healthy_loss'].item(),
                             gate_stage3_mean=gate_losses['gate_stage3_mean'].item(),
                             gate_stage4_mean=gate_losses['gate_stage4_mean'].item(),
                             lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, false_healthy_mask, has_false_healthy
        del seg_loss, gate_losses, loss, output, aux_outputs, data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    text_encoder_config = prepare_text_encoder_args(args)

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
        text_encoder = build_text_encoder(args, config=text_encoder_config)
        text_encoder = text_encoder.to(device)
        if args.distributed:
            text_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)
            text_encoder = torch.nn.parallel.DistributedDataParallel(text_encoder, device_ids=[args.local_rank])
        single_text_encoder = text_encoder.module if args.distributed else text_encoder
    else:
        text_encoder = None
        single_text_encoder = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        validate_checkpoint_config(checkpoint, args, args.resume, context='Resume')
        utils.load_state_dict_with_fallback(single_model,
                                            checkpoint['model'],
                                            strict=True,
                                            description='model')
        text_encoder_state, text_encoder_key = get_checkpoint_text_encoder_state(checkpoint, args)
        if args.model != 'lavt_one' and text_encoder_state is None:
            raise KeyError('Resume checkpoint [{}] is missing text encoder weights.'.format(args.resume))
        if args.model != 'lavt_one' and text_encoder_state is not None:
            utils.load_state_dict_with_fallback(single_text_encoder,
                                                text_encoder_state,
                                                strict=True,
                                                description=text_encoder_key)
        elif args.model == 'lavt_one' and text_encoder_state is not None:
            utils.load_state_dict_with_fallback(single_model.text_encoder,
                                                text_encoder_state,
                                                strict=True,
                                                description=text_encoder_key)

    backbone_no_decay = []
    backbone_decay = []
    for name, parameter in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    if args.model != 'lavt_one':
        text_encoder_params = [p for p in single_text_encoder.parameters() if p.requires_grad]
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": text_encoder_params},
        ]
    else:
        text_encoder_params = [p for p in single_model.text_encoder.parameters() if p.requires_grad]
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            {"params": text_encoder_params},
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

    resume_epoch = -999
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch,
                        args.print_freq, iterations, text_encoder, device, args)
        validation_result = evaluate(model, data_loader_val, text_encoder, device, args)
        current_score = validation_result['oIoU']

        if best_score < current_score:
            print('Better epoch: {}\n'.format(epoch))
            if args.model != 'lavt_one':
                dict_to_save = {
                    'model': single_model.state_dict(),
                    TEXT_ENCODER_MODEL_KEY: single_text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    CHECKPOINT_CONFIG_KEY: collect_checkpoint_config(args),
                    'lr_scheduler': lr_scheduler.state_dict()
                }
            else:
                dict_to_save = {
                    'model': single_model.state_dict(),
                    TEXT_ENCODER_MODEL_KEY: single_model.text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    CHECKPOINT_CONFIG_KEY: collect_checkpoint_config(args),
                    'lr_scheduler': lr_scheduler.state_dict()
                }

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_score = current_score

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser, validate_args

    parser = get_parser()
    args = validate_args(parser.parse_args())
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
