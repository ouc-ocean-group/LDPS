import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from .saliency_dataset import SaliencyDataset
from .dataloader import Dataloader
from torchvision.transforms import Compose
from lib.dataset.transform import ToTensor, Normalize, Resize, RandomHorizontalFlip

train_transform = Compose(
    [
        # Resize((256, 256)),
        # RandomHorizontalFlip(),
        ToTensor(),
        Normalize(),
    ]
)

test_transform = Compose([ToTensor(), Normalize()])


def make_dataloader(cfg, distributed=False, mode="train"):
    if mode == "train":
        dataset = SaliencyDataset(cfg.dataset, prior=cfg.prior, transform=train_transform)
        if cfg.gpu_id == 0:
            print(
                "=> [{}] Dataset: {} - Prior: {} | {} images.".format(
                    mode.upper(), cfg.dataset, cfg.prior, len(dataset)
                )
            )
    else:
        dataset = SaliencyDataset(cfg.test_dataset, prior=cfg.test_prior, transform=test_transform)
        if cfg.gpu_id == 0:
            print(
                "=> [{}] Dataset: {} - Prior: {} | {} images.".format(
                    mode.upper(), cfg.test_dataset, cfg.test_prior, len(dataset)
                )
            )

    data_sampler = DistributedSampler(dataset) if distributed else None
    data_loader_ = data.DataLoader(
        dataset,
        batch_size=cfg.batch_size if mode == "train" else 1,
        num_workers=cfg.num_workers,
        sampler=data_sampler,
        collate_fn=SaliencyDataset.collate_fn,
    )
    if mode == "train":
        data_loader = Dataloader(data_loader_, distributed)
    else:
        data_loader = data_loader_
        if distributed:
            data_sampler.set_epoch(0)

    return data_loader
