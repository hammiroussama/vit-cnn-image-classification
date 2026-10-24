"""Explicit experiment settings shared by training, evaluation and inference."""
import json
from dataclasses import asdict, dataclass

MODELS = ('cnn_scratch', 'vit', 'cnn', 'vit_small', 'vit_base')


@dataclass
class Config:
    dataset: str = 'cifar10'
    model_name: str = 'cnn_scratch'
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    num_classes: int = 10
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.0003
    backbone_lr_factor: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 15
    accumulation_steps: int = 1
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    augment: bool = True
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    seed: int = 42
    val_fraction: float = 0.1
    num_workers: int = 4
    amp: bool = True
    pretrained_id: str = ''

    @property
    def transfer(self):
        return self.model_name in ('cnn', 'vit_small', 'vit_base')

    @property
    def mixing(self):
        return self.augment and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

    def validate(self):
        if self.dataset not in ('cifar10', 'imagenette') or self.model_name not in MODELS:
            raise ValueError('Unsupported dataset or model')
        for key in ('epochs', 'batch_size', 'accumulation_steps', 'img_size', 'depth', 'embed_dim', 'num_heads', 'patch_size'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.num_classes != 10 or self.embed_dim % self.num_heads or self.img_size % self.patch_size:
            raise ValueError('Invalid class count or transformer dimensions')
        if not 0 < self.val_fraction < 1 or self.num_workers < 0 or self.lr <= 0:
            raise ValueError('Invalid split, workers or learning rate')
        if self.warmup_epochs < 0 or self.patience < 1 or self.backbone_lr_factor <= 0:
            raise ValueError('Invalid warmup, patience or backbone learning rate')
        if not 0 <= self.label_smoothing < 1 or min(self.mixup_alpha, self.cutmix_alpha) < 0:
            raise ValueError('Invalid smoothing or mixing setting')
        if any(not 0 <= v < 1 for v in (self.drop_rate, self.attn_drop_rate, self.drop_path_rate)):
            raise ValueError('Drop probabilities must be in [0, 1)')
        if self.transfer and self.img_size != 224:
            raise ValueError('Transfer baselines use 224px inputs and retain pretrained embeddings')
        if self.mixing and (self.batch_size < 2 or self.batch_size % 2):
            raise ValueError('MixUp/CutMix needs an even batch_size >= 2')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls(**json.load(f)).validate()
