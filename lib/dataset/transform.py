import torch
import random

from PIL import Image, ImageOps, ImageFilter

import torchvision.transforms.functional as F


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        img = F.normalize(samples[0], self.mean, self.std)
        return img, samples[1], samples[2]


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        img = samples[0].resize(self.size, Image.BILINEAR)
        prior = samples[1].resize(self.size, Image.NEAREST)
        gt = samples[2].resize(self.size, Image.NEAREST)
        return img, prior, gt


class ToTensor(object):
    def __call__(self, samples):
        img = F.to_tensor(samples[0])
        prior = F.to_tensor(samples[1])
        gt = F.to_tensor(samples[2])
        if prior.size(0) != 1:
            prior = prior[0:1]
        if gt.size(0) != 1:
            gt = gt[0:1]
        return img, prior, gt


class RandomHorizontalFlip(object):
    def __call__(self, samples):
        new_samples = []
        if random.random() < 0.5:
            for i, item in enumerate(samples):
                new_samples.append(item.transpose(Image.FLIP_LEFT_RIGHT))
            return new_samples
        else:
            return samples


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        sample["image"] = img
        sample["label"] = mask

        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        sample["image"] = img
        sample["label"] = mask

        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, samples):
        img = samples[0]
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        for i, item in enumerate(samples):
            if samples[i].mode == "RGB":
                samples[i] = item.resize((ow, oh), Image.BILINEAR)
            else:
                samples[i] = item.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            for i, item in enumerate(samples):
                samples[i] = ImageOps.expand(item, border=(0, 0, padw, padh), fill=self.fill[i])
        # random crop crop_size
        w, h = samples[0].size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        for i, item in enumerate(samples):
            samples[i] = item.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return samples


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.0))
        y1 = int(round((h - self.crop_size) / 2.0))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, samples):
        w, h = samples[0].size

        long_side = max(w, h)

        ratio = self.size / long_side
        if w < h:
            oh = self.size
            ow = int(ratio * w)
        else:
            ow = self.size
            oh = int(ratio * h)

        for i, item in enumerate(samples):
            if samples[i].mode == "RGB":
                samples[i] = item.resize((ow, oh), Image.BILINEAR)
            else:
                samples[i] = item.resize((ow, oh), Image.NEAREST)

        return samples
