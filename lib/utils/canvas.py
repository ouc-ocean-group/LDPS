import torchvision.utils as vutils
import numpy as np
import torch
from lib.utils.transform import denorm, to_img, to_tensor


def hstack(imgs):
    out = imgs[0]
    for i in range(1, imgs.shape[0]):
        out = np.hstack((out, imgs[i]))
    return out


def vstack(imgs):
    out = imgs[0]
    for i in range(1, len(imgs)):
        out = np.vstack((out, imgs[i]))
    return out


def stack(imgs):
    out = []
    for img in imgs:
        img = hstack(img)
        out.append(img)
    out = vstack(out)
    return out


def make_img(imgs, ifdenorm=True):
    img_list = []
    for img in imgs:
        img = img.numpy().transpose((0, 2, 3, 1))
        img_list.append(img)

    result = stack(img_list)

    log_img = torch.from_numpy(result).float().permute(2, 0, 1)

    if ifdenorm:
        log_img[:, 0 : int(log_img.size(1) / 5), :] = denorm(log_img[:, 0 : int(log_img.size(1) / 5), :])
    else:
        log_img[:, 0 : int(log_img.size(1) / 4), :] = denorm(log_img[:, 0 : int(log_img.size(1) / 4), :])
    return vutils.make_grid(log_img, normalize=False, scale_each=True)
