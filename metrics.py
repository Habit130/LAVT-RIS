import numpy as np
import torch


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_confusion_matrix(prediction, target, num_classes=2):
    prediction = prediction.to(torch.int64)
    target = target.to(torch.int64)
    valid_mask = (target >= 0) & (target < num_classes)
    encoded = target[valid_mask] * num_classes + prediction[valid_mask]
    confusion = torch.bincount(encoded.view(-1), minlength=num_classes ** 2)
    return confusion.reshape(num_classes, num_classes)


class BinarySegmentationMeter(object):
    def __init__(self):
        self.confusion = np.zeros((2, 2), dtype=np.int64)

    def update_from_logits(self, logits, target):
        prediction = logits.argmax(1)
        confusion = _to_confusion_matrix(prediction.detach().cpu(), target.detach().cpu(), num_classes=2)
        self.confusion += confusion.numpy()

    def compute(self):
        tn, fp = self.confusion[0, 0], self.confusion[0, 1]
        fn, tp = self.confusion[1, 0], self.confusion[1, 1]

        fg_iou = _safe_divide(tp, tp + fp + fn)
        dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
        recall = _safe_divide(tp, tp + fn)
        bg_iou = _safe_divide(tn, tn + fp + fn)
        bg_acc = _safe_divide(tn, tn + fp)

        return {
            'fg_iou': fg_iou,
            'dice': dice,
            'recall': recall,
            'miou': (fg_iou + bg_iou) / 2.0,
            'macc': (recall + bg_acc) / 2.0,
        }


def format_binary_metrics(metrics_dict):
    return '\n'.join([
        'IoU: {:.2f}'.format(metrics_dict['fg_iou'] * 100.0),
        'Dice: {:.2f}'.format(metrics_dict['dice'] * 100.0),
        'Recall: {:.2f}'.format(metrics_dict['recall'] * 100.0),
        'mIoU: {:.2f}'.format(metrics_dict['miou'] * 100.0),
        'mACC: {:.2f}'.format(metrics_dict['macc'] * 100.0),
    ])
