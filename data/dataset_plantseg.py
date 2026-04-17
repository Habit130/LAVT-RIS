import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

from text_encoder import build_text_tokenizer, tokenize_text


class PlantSegDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.eval_mode = eval_mode
        self.max_tokens = args.max_text_tokens
        self.caption_index = args.plantseg_caption_index
        self.root = Path(args.plantseg_root).expanduser().resolve()
        self.metadata_path = self.root / 'main.json'

        with self.metadata_path.open('r', encoding='utf-8') as handle:
            records = json.load(handle)

        self.samples = [record for record in records if record.get('split') == split]
        if not self.samples:
            raise ValueError('No plantseg samples found for split [{}] under {}'.format(split, self.metadata_path))

        self.tokenizer = build_text_tokenizer(args)
        self.input_ids = []
        self.attention_masks = []

        for sample in self.samples:
            captions = sample.get('caption') or []
            if len(captions) <= self.caption_index:
                raise IndexError(
                    'Sample [{}] does not have caption index {}'.format(sample.get('id', '<unknown>'),
                                                                        self.caption_index)
                )

            sentence_raw = captions[self.caption_index]
            input_ids, attention_mask = tokenize_text(self.tokenizer, sentence_raw, self.max_tokens)

            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)

    @staticmethod
    def _mask_to_tensor(mask):
        mask = mask.convert('L')
        tensor = torch.frombuffer(mask.tobytes(), dtype=torch.uint8)
        tensor = tensor.view(mask.size[1], mask.size[0]).clone().contiguous()
        return tensor.gt(0).to(dtype=torch.int64)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample['image']
        mask_path = self.root / sample['mask']
        false_healthy_rel = sample.get('false_healthy_ann')

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        target = mask.point(lambda pixel: 1 if pixel > 0 else 0, mode='L')
        if false_healthy_rel:
            false_healthy_mask = Image.open(self.root / false_healthy_rel).convert('L')
        else:
            false_healthy_mask = Image.new('L', mask.size, 0)

        if self.image_transforms is not None:
            image, target = self.image_transforms(image, target)
            if isinstance(target, torch.Tensor):
                false_healthy_mask = self._mask_to_tensor(false_healthy_mask).unsqueeze(0).unsqueeze(0).float()
                false_healthy_mask = F.interpolate(false_healthy_mask,
                                                   size=target.shape[-2:],
                                                   mode='nearest').squeeze(0).squeeze(0).to(dtype=torch.int64)
            else:
                false_healthy_mask = false_healthy_mask.resize((target.size[0], target.size[1]), resample=Image.NEAREST)
        else:
            false_healthy_mask = self._mask_to_tensor(false_healthy_mask)

        sentence = self.input_ids[index]
        attention_mask = self.attention_masks[index]

        if self.eval_mode:
            sentence = sentence.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)
            return image, target, sentence, attention_mask, sample['mask']

        has_false_healthy = torch.tensor(1 if false_healthy_rel else 0, dtype=torch.float32)
        return image, target, sentence, attention_mask, false_healthy_mask, has_false_healthy
