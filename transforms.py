from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)  # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        image = F.resize(image, size)
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    @staticmethod
    def _pil_image_to_tensor(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        channels = len(image.getbands())
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = tensor.view(image.size[1], image.size[0], channels)
        tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor.float().div(255.0)

    @staticmethod
    def _pil_mask_to_tensor(mask):
        mask = mask.convert("L")
        tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
        tensor = tensor.view(mask.size[1], mask.size[0]).contiguous()
        return tensor.gt(0).to(dtype=torch.int64)

    def __call__(self, image, target):
        image = self._pil_image_to_tensor(image)
        target = self._pil_mask_to_tensor(target)
        return image, target


class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

