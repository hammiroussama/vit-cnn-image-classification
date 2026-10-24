from torchvision import transforms as T
from timm.data import resolve_data_config
from timm.data.mixup import Mixup


def preprocessing(cfg, model=None):
    if cfg.model_name in ('vit_small', 'vit_base'):
        resolved = resolve_data_config({}, model=model)
        return {'size': cfg.img_size, 'mean': list(resolved['mean']),
                'std': list(resolved['std']), 'interpolation': resolved['interpolation'],
                'resize': int(cfg.img_size / resolved.get('crop_pct', 0.9)),
                'native_cifar': False}
    native = cfg.dataset == 'cifar10' and not cfg.transfer
    return {'size': cfg.img_size, 'mean': [0.4914, 0.4822, 0.4465] if native else [0.485, 0.456, 0.406],
            'std': [0.2470, 0.2435, 0.2616] if native else [0.229, 0.224, 0.225],
            'interpolation': 'bilinear', 'resize': 256 if not native else cfg.img_size,
            'native_cifar': native}


def transform(spec, train=False, augment=True):
    interpolation = T.InterpolationMode(spec['interpolation'])
    if train and augment:
        ops = ([T.RandomCrop(spec['size'], padding=4)] if spec['native_cifar'] else
               [T.RandomResizedCrop(spec['size'], interpolation=interpolation)])
        ops += [T.RandomHorizontalFlip()]
    elif spec['native_cifar']:
        ops = [T.Resize((spec['size'], spec['size']), interpolation=interpolation)]
    else:
        ops = [T.Resize(spec['resize'], interpolation=interpolation), T.CenterCrop(spec['size'])]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(spec['mean'], spec['std'])])


def mixing(cfg):
    return Mixup(mixup_alpha=cfg.mixup_alpha, cutmix_alpha=cfg.cutmix_alpha,
                 label_smoothing=cfg.label_smoothing, num_classes=cfg.num_classes) if cfg.mixing else None
