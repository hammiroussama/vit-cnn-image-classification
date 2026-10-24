import torch.nn as nn
import timm
from torchvision.models import resnet18, ResNet18_Weights
from .custom_vit import ViT


def build_model(cfg, initialize_pretrained=True):
    """Inference never downloads weights; checkpoint loading is strict."""
    cfg.validate()
    if cfg.model_name in ('cnn', 'cnn_scratch'):
        weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.transfer and initialize_pretrained else None
        model = resnet18(weights=weights)
        if cfg.dataset == 'cifar10' and not cfg.transfer:
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
        return model
    if cfg.model_name == 'vit':
        return ViT(img_size=cfg.img_size, patch_size=cfg.patch_size,
                   embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads,
                   num_classes=cfg.num_classes, qkv_bias=True,
                   drop_rate=cfg.drop_rate, attn_drop_rate=cfg.attn_drop_rate,
                   drop_path_rate=cfg.drop_path_rate)
    if not cfg.pretrained_id:
        raise ValueError('A specific timm pretrained_id is required')
    # Intentionally no fallback: missing or incompatible pretrained weights fail the run.
    return timm.create_model(cfg.pretrained_id, pretrained=initialize_pretrained,
                             num_classes=cfg.num_classes, drop_rate=cfg.drop_rate,
                             attn_drop_rate=cfg.attn_drop_rate, drop_path_rate=cfg.drop_path_rate)


def parameter_groups(model, cfg):
    if not cfg.transfer:
        return [{'params': list(model.parameters()), 'lr': cfg.lr}]
    head = model.fc if cfg.model_name == 'cnn' else model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    return [
        {'params': [p for p in model.parameters() if id(p) not in head_ids],
         'lr': cfg.lr * cfg.backbone_lr_factor},
        {'params': list(head.parameters()), 'lr': cfg.lr},
    ]
