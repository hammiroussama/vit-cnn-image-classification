import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageFolder
from data_aug import transform

IMAGENETTE = dict(zip(
    ['n01440764','n02102040','n02979186','n03000684','n03028079',
     'n03394916','n03417042','n03425413','n03445777','n03888257'],
    ['tench','English springer spaniel','cassette player','chain saw','church',
     'French horn','garbage truck','gas pump','golf ball','parachute']))


def stratified_split(labels, fraction, seed):
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = [i for i, target in enumerate(labels) if target == label]
        if len(indices) < 2:
            raise ValueError('Each class needs at least two training examples')
        rng.shuffle(indices)
        n = min(len(indices)-1, max(1, round(len(indices) * fraction)))
        val.extend(indices[:n]); train.extend(indices[n:])
    return sorted(train), sorted(val)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def dataset(cfg, root, spec, split, augment=False, download=False):
    tf = transform(spec, train=augment, augment=cfg.augment)
    if cfg.dataset == 'cifar10':
        return CIFAR10(root, train=split != 'test', download=download, transform=tf)
    return ImageFolder(Path(root) / ('val' if split == 'test' else 'train'), transform=tf)


def training_loaders(cfg, root, spec, download=False):
    # The official test/Imagenette validation directory is never loaded here.
    train_data = dataset(cfg, root, spec, 'train', augment=True, download=download)
    val_data = dataset(cfg, root, spec, 'train', download=download)
    if len(train_data.classes) != cfg.num_classes:
        raise ValueError('Dataset class count does not match the configuration')
    if cfg.dataset == 'imagenette' and set(train_data.classes) != set(IMAGENETTE):
        raise ValueError('Expected the ten canonical Imagenette class folders')
    train_ids, val_ids = stratified_split(train_data.targets, cfg.val_fraction, cfg.seed)
    if cfg.mixing and len(train_ids) < cfg.batch_size:
        raise ValueError('Training subset is smaller than one full MixUp batch')
    kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0, worker_init_fn=worker_seed,
                  pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(Subset(train_data, train_ids), shuffle=True, generator=generator,
                              drop_last=cfg.mixing, **kwargs)
    val_loader = DataLoader(Subset(val_data, val_ids), shuffle=False, **kwargs)
    classes = [IMAGENETTE.get(c, c) for c in train_data.classes]
    manifest = {'train_indices': train_ids, 'validation_indices': val_ids,
                'class_to_idx': train_data.class_to_idx, 'seed': cfg.seed,
                'source_split': 'training', 'validation_fraction': cfg.val_fraction}
    if cfg.dataset == 'imagenette':
        manifest['ordered_files'] = [Path(p).relative_to(Path(root) / 'train').as_posix()
                                     for p, _ in train_data.samples]
    return train_loader, val_loader, classes, manifest
