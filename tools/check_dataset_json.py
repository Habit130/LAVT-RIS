import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_path(dataset_root, value):
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def normalize_captions(sample):
    for key in ('caption', 'captions', 'sentence', 'sentences'):
        if key in sample:
            value = sample[key]
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
            raise TypeError(f"Sample {sample.get('id', '<unknown>')} has unsupported caption type {type(value).__name__}")
    raise KeyError(f"Sample {sample.get('id', '<unknown>')} has no caption field")


def inspect_split(dataset_root, split_name, json_name, max_mask_checks):
    json_path = (dataset_root / json_name).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f'Split file not found: {json_path}')

    with json_path.open('r', encoding='utf-8') as handle:
        samples = json.load(handle)

    if not isinstance(samples, list):
        raise ValueError(f'Expected a list in {json_path}, got {type(samples).__name__}')

    missing_paths = []
    mask_values = set()
    foreground_counts = []

    for index, sample in enumerate(samples):
        image_path = resolve_path(dataset_root, sample['image'])
        mask_path = resolve_path(dataset_root, sample['mask'])
        normalize_captions(sample)

        if not image_path.is_file():
            missing_paths.append(str(image_path))
        if not mask_path.is_file():
            missing_paths.append(str(mask_path))

        if index < max_mask_checks and mask_path.is_file():
            mask = np.array(Image.open(mask_path))
            mask_values.update(int(value) for value in np.unique(mask))
            foreground_counts.append(int((mask > 0).sum()))

    return {
        'split': split_name,
        'json_path': str(json_path),
        'count': len(samples),
        'missing_paths': missing_paths,
        'sample_mask_values': sorted(mask_values),
        'sample_foreground_pixels': foreground_counts[:5],
    }


def main():
    parser = argparse.ArgumentParser(description='Check local LAVT-RIS JSON dataset wiring')
    parser.add_argument('--dataset_root', default='../dataset', help='dataset root containing train/test JSON files')
    parser.add_argument('--train_json', default='train.json', help='train split JSON filename')
    parser.add_argument('--test_json', default='test.json', help='test split JSON filename')
    parser.add_argument('--max_mask_checks', default=64, type=int, help='number of masks per split used to inspect values')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f'Dataset root not found: {dataset_root}')

    reports = [
        inspect_split(dataset_root, 'train', args.train_json, args.max_mask_checks),
        inspect_split(dataset_root, 'test', args.test_json, args.max_mask_checks),
    ]

    print(f'dataset_root={dataset_root}')
    for report in reports:
        print(f"[{report['split']}] json={report['json_path']}")
        print(f"[{report['split']}] samples={report['count']}")
        print(f"[{report['split']}] missing_paths={len(report['missing_paths'])}")
        if report['missing_paths']:
            print(f"[{report['split']}] first_missing={report['missing_paths'][0]}")
        print(f"[{report['split']}] sample_mask_values={report['sample_mask_values']}")
        print(f"[{report['split']}] sample_foreground_pixels={report['sample_foreground_pixels']}")


if __name__ == '__main__':
    main()
