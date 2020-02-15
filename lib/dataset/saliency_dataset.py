import torch.utils.data as data
from PIL import Image
import os
import torch
import numpy as np
import os.path as osp
from torchvision.transforms.functional import to_tensor


class SaliencyDataset(data.Dataset):
    def __init__(self, dataset, prior, transform):
        self._data_path = "./data/{}".format(dataset)
        self._prior_path = "{}/{}".format(self._data_path, prior)
        self.image_index = self._load_image_set_index()
        print("=> Checking prior...")
        self.image_index = self._prior_check()
        print("=> Done.")
        self.transform = transform

    def loader(self, item_idx):
        img_path = osp.join(self._data_path, "imgs", self.image_index[item_idx][0])
        prior_path = osp.join(self._prior_path, self.image_index[item_idx][1])
        gt_path = osp.join(self._data_path, "gt", self.image_index[item_idx][2])

        img = Image.open(img_path).convert("RGB")
        prior = Image.open(prior_path)
        gt = Image.open(gt_path)

        if prior.size != img.size:
            prior = prior.resize(img.size, Image.ANTIALIAS)
        if gt.size != img.size:
            gt = gt.resize(img.size, Image.ANTIALIAS)
        return img, prior, gt

    def __getitem__(self, item_idx):
        sample = self.loader(item_idx)
        sample = self.transform(sample)
        return (*sample, self.image_index[item_idx][0])

    @staticmethod
    def collate_fn(batch):
        imgs, priors, gts, names = [], [], [], []
        for batch_block in batch:
            imgs.append(batch_block[0].unsqueeze(0))
            priors.append(batch_block[1].unsqueeze(0))
            gts.append(batch_block[2].unsqueeze(0))
            names.append(batch_block[3])

        imgs, priors, gts = torch.cat(imgs), torch.cat(priors), torch.cat(gts)
        return imgs, priors, gts, names

    def __len__(self):
        return len(self.image_index)

    def _load_image_set_index(self):
        img_idx = [x.split(".")[0] for x in os.listdir(os.path.join(self._data_path, "imgs"))]
        prior_idx = [x.split(".")[0] for x in os.listdir(self._prior_path)]
        gt_idx = [x.split(".")[0] for x in os.listdir(os.path.join(self._data_path, "gt"))]

        idx = list(set(img_idx) & set(prior_idx) & set(gt_idx))
        idx = list(np.sort(idx))
        imgs = [x + ".jpg" for x in idx]
        priors = [x + ".png" for x in idx]
        gts = [x + ".png" for x in idx]
        image_index = list(zip(imgs, priors, gts))
        return image_index

    def _prior_check(self):
        new_image_idx = []
        for image_name, prior_name, gt_name in self.image_index:
            prior_path = osp.join(self._prior_path, prior_name)
            prior = Image.open(prior_path)
            prior = to_tensor(prior)
            if prior.max() < 0.9 or prior.min() > 0.3:
                continue
            new_image_idx.append([image_name, prior_name, gt_name])
        return new_image_idx
