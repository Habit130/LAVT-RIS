import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

from bert.tokenization_bert import BertTokenizer


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = ['background', 'foreground']
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode
        self.caption_index = args.caption_index
        self.max_tokens = args.text_max_tokens
        self.dataset_root = Path(args.dataset_root).expanduser().resolve()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        split_files = {
            'train': args.train_json,
            'test': args.test_json,
        }
        if self.split not in split_files:
            raise ValueError(
                f"Unsupported split '{self.split}'. Available splits: {sorted(split_files)}"
            )

        annotations_path = self.dataset_root / split_files[self.split]
        if not annotations_path.is_file():
            raise FileNotFoundError(f'Annotation file not found: {annotations_path}')

        with annotations_path.open('r', encoding='utf-8') as handle:
            samples = json.load(handle)

        if not isinstance(samples, list):
            raise ValueError(f'Expected a list of samples in {annotations_path}, got {type(samples).__name__}')

        self.samples = []
        self.input_ids = []
        self.attention_masks = []

        for sample in samples:
            image_path = self._resolve_path(sample.get('image'))
            mask_path = self._resolve_path(sample.get('mask'))
            captions = self._normalize_captions(sample)
            sample_id = sample.get('id', image_path.stem)
            selected_caption = self._select_caption(captions, sample_id)

            self.samples.append({
                'id': sample_id,
                'image_path': image_path,
                'mask_path': mask_path,
                'caption': selected_caption,
            })

            token_ids = self.tokenizer.encode(text=selected_caption, add_special_tokens=True)
            token_ids = token_ids[:self.max_tokens]

            padded_input_ids = [0] * self.max_tokens
            attention_mask = [0] * self.max_tokens
            padded_input_ids[:len(token_ids)] = token_ids
            attention_mask[:len(token_ids)] = [1] * len(token_ids)

            self.input_ids.append(torch.tensor(padded_input_ids).unsqueeze(0))
            self.attention_masks.append(torch.tensor(attention_mask).unsqueeze(0))

    def _normalize_captions(self, sample):
        raw_captions = sample.get('caption')
        if raw_captions is None:
            raw_captions = sample.get('captions')
        if raw_captions is None:
            raw_captions = sample.get('sentence')
        if raw_captions is None:
            raw_captions = sample.get('sentences')
        if raw_captions is None:
            raise KeyError(f"Sample {sample.get('id', '<unknown>')} does not contain a caption field")

        if isinstance(raw_captions, str):
            captions = [raw_captions]
        elif isinstance(raw_captions, list):
            captions = [str(caption) for caption in raw_captions if str(caption).strip()]
        else:
            raise TypeError(
                f"Sample {sample.get('id', '<unknown>')} has unsupported caption type {type(raw_captions).__name__}"
            )

        if not captions:
            raise ValueError(f"Sample {sample.get('id', '<unknown>')} does not contain a usable caption")

        return captions

    def _select_caption(self, captions, sample_id):
        try:
            return captions[self.caption_index]
        except IndexError as exc:
            raise IndexError(
                f"Sample {sample_id} has {len(captions)} captions, but caption_index={self.caption_index} is out of range"
            ) from exc

    def _resolve_path(self, path_value):
        if not path_value:
            raise KeyError(f"Split '{self.split}' contains a sample without image/mask path")

        path = Path(path_value)
        if not path.is_absolute():
            path = self.dataset_root / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f'File referenced by split {self.split} not found: {path}')

        return path

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample['image_path']).convert('RGB')
        mask = np.array(Image.open(sample['mask_path']))
        mask = (mask > 0).astype(np.uint8)
        target = Image.fromarray(mask, mode='L')

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        tensor_embeddings = self.input_ids[index]
        attention_mask = self.attention_masks[index]

        if self.eval_mode:
            tensor_embeddings = tensor_embeddings.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)

        return image, target, tensor_embeddings, attention_mask
