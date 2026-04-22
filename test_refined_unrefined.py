import torch
import torch.utils.data

from eval_ris_metrics import evaluate_mask_arrays, format_metrics_summary
from lib import segmentation
from test import (allow_partial_checkpoint_load, forward_model, get_dataset,
                  get_transform)
from text_encoder import (build_text_encoder, get_checkpoint_text_encoder_state,
                          prepare_text_encoder_args)
import utils


def _build_model_and_text_encoder(args, device):
    text_encoder_config = prepare_text_encoder_args(args)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    utils.load_state_dict_with_fallback(single_model,
                                        checkpoint['model'],
                                        strict=not allow_partial_checkpoint_load(args),
                                        description='model')
    model = single_model.to(device)

    if args.model != 'lavt_one':
        single_text_encoder = build_text_encoder(args, config=text_encoder_config)
        text_encoder_state, text_encoder_key = get_checkpoint_text_encoder_state(checkpoint, args)
        if text_encoder_state is not None:
            utils.load_state_dict_with_fallback(single_text_encoder,
                                                text_encoder_state,
                                                strict=True,
                                                description=text_encoder_key)
        text_encoder = single_text_encoder.to(device)
    else:
        text_encoder = None

    return model, text_encoder


def evaluate_refined_groups(model, data_loader, dataset, text_encoder, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test by refine group:'

    refined_predictions = []
    refined_targets = []
    unrefined_predictions = []
    unrefined_targets = []
    refined_count = 0
    unrefined_count = 0

    with torch.no_grad():
        for index, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, sentences, attentions, _mask_paths = data
            sample = dataset.samples[index]
            is_refined = bool(sample.get('is_refined', False))

            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device).squeeze(1)
            attentions = attentions.to(device).squeeze(1)

            for sentence_idx in range(sentences.size(-1)):
                output = forward_model(model, text_encoder, image, sentences[:, :, sentence_idx], attentions[:, :, sentence_idx])
                prediction = output.argmax(1).detach().cpu().tolist()
                target_list = target.detach().cpu().tolist()
                if is_refined:
                    refined_predictions.extend(prediction)
                    refined_targets.extend(target_list)
                else:
                    unrefined_predictions.extend(prediction)
                    unrefined_targets.extend(target_list)

            if is_refined:
                refined_count += 1
            else:
                unrefined_count += 1

    print('Refined samples: {}'.format(refined_count))
    if refined_predictions:
        print(format_metrics_summary(evaluate_mask_arrays(refined_predictions, refined_targets)))
    else:
        print('No refined samples available for evaluation.')
    print('')
    print('Unrefined samples: {}'.format(unrefined_count))
    if unrefined_predictions:
        print(format_metrics_summary(evaluate_mask_arrays(unrefined_predictions, unrefined_targets)))
    else:
        print('No unrefined samples available for evaluation.')


def main(args):
    if args.dataset != 'plantseg':
        raise ValueError('test_refined_unrefined.py only supports --dataset plantseg')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    prepare_text_encoder_args(args)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=args.workers)

    model, text_encoder = _build_model_and_text_encoder(args, device)
    evaluate_refined_groups(model, data_loader_test, dataset_test, text_encoder, device)


if __name__ == "__main__":
    from args import get_parser, validate_args

    parser = get_parser()
    args = validate_args(parser.parse_args())
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
