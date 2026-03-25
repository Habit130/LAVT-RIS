import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_binary_stats(pred_seg, gt_seg):
    pred_fg = pred_seg.astype(bool)
    gt_fg = gt_seg.astype(bool)

    tp = np.logical_and(pred_fg, gt_fg).sum(dtype=np.int64)
    fp = np.logical_and(pred_fg, np.logical_not(gt_fg)).sum(dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred_fg), gt_fg).sum(dtype=np.int64)
    tn = np.logical_and(np.logical_not(pred_fg), np.logical_not(gt_fg)).sum(dtype=np.int64)

    return tp, fp, fn, tn


def summarize_metrics(tp, fp, fn, tn):
    iou_fg = safe_divide(tp, tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    recall = safe_divide(tp, tp + fn)
    iou_bg = safe_divide(tn, tn + fn + fp)
    m_iou = (iou_fg + iou_bg) / 2.0
    acc_fg = safe_divide(tp, tp + fn)
    acc_bg = safe_divide(tn, tn + fp)
    m_acc = (acc_fg + acc_bg) / 2.0

    return {
        'IoU': iou_fg,
        'Dice': dice,
        'Recall': recall,
        'mIoU': m_iou,
        'mACC': m_acc,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
    }


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target_np = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                tp, fp, fn, tn = compute_binary_stats(output_mask, target_np)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    metrics = summarize_metrics(total_tp, total_fp, total_fn, total_tn)
    print('Final results:')
    print('    TP = {}'.format(metrics['TP']))
    print('    FP = {}'.format(metrics['FP']))
    print('    FN = {}'.format(metrics['FN']))
    print('    TN = {}'.format(metrics['TN']))
    print('    IoU = {:.2f}'.format(metrics['IoU'] * 100.0))
    print('    Dice = {:.2f}'.format(metrics['Dice'] * 100.0))
    print('    Recall = {:.2f}'.format(metrics['Recall'] * 100.0))
    print('    mIoU = {:.2f}'.format(metrics['mIoU'] * 100.0))
    print('    mACC = {:.2f}'.format(metrics['mACC'] * 100.0))



def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


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
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    checkpoint = torch.load(args.resume, map_location='cpu')
    sync_decoder_args_from_checkpoint(args, checkpoint.get('args'))
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
