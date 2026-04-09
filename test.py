from pathlib import Path

import torch
import torch.utils.data
from PIL import Image

from bert.modeling_bert import BertModel

from lib import segmentation
import metrics
import transforms as T
import utils
import numpy as np


def get_dataset(image_set, transform, args):
    if args.dataset == 'plantseg':
        from data.dataset_plantseg import PlantSegDataset
        ds = PlantSegDataset(args,
                             split=image_set,
                             image_transforms=transform,
                             target_transforms=None,
                             eval_mode=True)
    else:
        from data.dataset_refer_bert import ReferDataset
        ds = ReferDataset(args,
                          split=image_set,
                          image_transforms=transform,
                          target_transforms=None,
                          eval_mode=True)
    num_classes = 2
    return ds, num_classes


def forward_model(model, bert_model, image, sentences, attentions):
    if bert_model is not None:
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
        embedding = last_hidden_states.permute(0, 2, 1)
        return model(image, embedding, l_mask=attentions.unsqueeze(-1))
    return model(image, sentences, l_mask=attentions)


def evaluate(model, data_loader, bert_model, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test:'

    if args.dataset == 'plantseg':
        meter = metrics.BinarySegmentationMeter()
        save_root = Path(args.save_mask_dir).expanduser().resolve() if args.save_mask_dir else None
        with torch.no_grad():
            for data in metric_logger.log_every(data_loader, 100, header):
                image, target, sentences, attentions, mask_paths = data
                image = image.to(device)
                target = target.to(device)
                sentences = sentences.to(device).squeeze(1)
                attentions = attentions.to(device).squeeze(1)

                for j in range(sentences.size(-1)):
                    output = forward_model(model, bert_model, image, sentences[:, :, j], attentions[:, :, j])
                    meter.update_from_logits(output, target)
                    if save_root is not None:
                        save_prediction_mask(output, mask_paths[0], args, save_root)

        print('Final results:')
        print(metrics.format_binary_metrics(meter.compute()))
        return

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                output = forward_model(model, bert_model, image, sentences[:, :, j], attentions[:, :, j])
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    overall_iou = float(cum_I * 100. / cum_U) if cum_U != 0 else 0.0
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % overall_iou
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def _prediction_tensor_to_bytes(prediction):
    prediction = prediction.detach().cpu().to(torch.uint8).mul(255).contiguous().view(-1)
    return bytes(prediction.tolist())


def save_prediction_mask(logits, mask_relative_path, args, save_root):
    prediction = logits.argmax(1)[0]
    flat_bytes = _prediction_tensor_to_bytes(prediction)
    mask_image = Image.frombytes('L', (prediction.shape[1], prediction.shape[0]), flat_bytes)

    reference_mask_path = Path(args.plantseg_root).expanduser().resolve() / mask_relative_path
    with Image.open(reference_mask_path) as reference_mask:
        if mask_image.size != reference_mask.size:
            mask_image = mask_image.resize(reference_mask.size, resample=Image.NEAREST)
        if reference_mask.mode != mask_image.mode:
            mask_image = mask_image.convert(reference_mask.mode)

    output_path = save_root / mask_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(output_path)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
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

    evaluate(model, data_loader_test, bert_model, device=device, args=args)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
