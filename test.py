import json
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image

from lib import segmentation
from text_encoder import (build_text_encoder, encode_text, get_checkpoint_text_encoder_state,
                          prepare_text_encoder_args)
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
            '{} checkpoint [{}] is missing [{}]; use a new-format checkpoint or convert it after '
            'manually confirming the config.'
            .format(context, checkpoint_path, CHECKPOINT_CONFIG_KEY)
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


def forward_model(model, text_encoder, image, sentences, attentions):
    if text_encoder is not None:
        last_hidden_states = encode_text(text_encoder, sentences, attentions)
        embedding = last_hidden_states.permute(0, 2, 1)
        return model(image, embedding, l_mask=attentions.unsqueeze(-1))
    return model(image, sentences, l_mask=attentions)


def save_predictions(model, data_loader, text_encoder, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test:'
    save_root = Path(args.save_pred_dir).expanduser().resolve()

    if args.dataset != 'plantseg':
        raise ValueError('Saving prediction masks currently requires dataset metadata with GT mask relative paths; '
                         '--dataset plantseg is supported in test.py.')

    save_count = 0
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, _target, sentences, attentions, mask_paths = data
            image = image.to(device)
            sentences = sentences.to(device).squeeze(1)
            attentions = attentions.to(device).squeeze(1)

            for j in range(sentences.size(-1)):
                output = forward_model(model, text_encoder, image, sentences[:, :, j], attentions[:, :, j])
                save_prediction_mask(output, mask_paths[0], args, save_root)
                save_count += 1

            del image, sentences, attentions, output

    print('Saved {} prediction masks to {}'.format(save_count, save_root))


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


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

    gt_mask_root = Path(args.plantseg_root).expanduser().resolve() / 'ann'
    try:
        relative_output_path = reference_mask_path.relative_to(gt_mask_root)
    except ValueError:
        relative_output_path = Path(mask_relative_path)

    output_path = save_root / relative_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(output_path)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if not args.save_pred_dir:
        raise ValueError('test.py now only generates prediction masks; please specify --save_pred_dir.')
    text_encoder_config = prepare_text_encoder_args(args)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    validate_checkpoint_config(checkpoint, args, args.resume, context='Test')
    utils.load_state_dict_with_fallback(single_model,
                                        checkpoint['model'],
                                        strict=True,
                                        description='model')
    model = single_model.to(device)

    if args.model != 'lavt_one':
        single_text_encoder = build_text_encoder(args, config=text_encoder_config)
        text_encoder_state, text_encoder_key = get_checkpoint_text_encoder_state(checkpoint, args)
        if text_encoder_state is None:
            raise KeyError('Test checkpoint [{}] is missing text encoder weights.'.format(args.resume))
        if text_encoder_state is not None:
            utils.load_state_dict_with_fallback(single_text_encoder,
                                                text_encoder_state,
                                                strict=True,
                                                description=text_encoder_key)
        text_encoder = single_text_encoder.to(device)
    else:
        text_encoder = None

    save_predictions(model, data_loader_test, text_encoder, device=device, args=args)


if __name__ == "__main__":
    from args import get_parser, validate_args
    parser = get_parser()
    args = validate_args(parser.parse_args())
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
