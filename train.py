"""Train on a development split. Final testing is a separate command."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from models.factory import build_model, parameter_groups
from data_aug import preprocessing, mixing
from data import training_loaders
from metrics import save_json, classification_metrics, plot_history


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, scaler, cfg, device, mix_fn=None):
    model.train(); optimizer.zero_grad(set_to_none=True)
    total_loss = 0.; samples = 0; window_samples = 0; updates = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if mix_fn is not None: images, labels = mix_fn(images, labels)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.amp and device.type == 'cuda'):
            logits = model(images)
            # MixUp has already smoothed soft targets.
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0 if mix_fn else cfg.label_smoothing)
        n = len(images)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        scaler.scale(loss * n).backward()
        window_samples += n; samples += n; total_loss += loss.item() * n
        if (i + 1) % cfg.accumulation_steps == 0 or i + 1 == len(loader):
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(window_samples)
            if cfg.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            window_samples = 0; updates += 1
    if samples == 0: raise ValueError('No training samples processed')
    return total_loss / samples, updates


@torch.inference_mode()
def evaluate(model, loader, device, class_count=10):
    model.eval(); truth = []; predictions = []; loss_sum = 0.
    for images, labels in loader:
        logits = model(images.to(device))
        if not torch.isfinite(logits).all(): raise RuntimeError('Non-finite predictions')
        loss_sum += F.cross_entropy(logits, labels.to(device), reduction='sum').item()
        truth.extend(labels.tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
    result = classification_metrics(truth, predictions, class_count)
    result['loss'] = loss_sum / len(truth)
    return result, truth, predictions


def run(cfg, data_root, out, device, download=False):
    cfg.validate(); set_seed(cfg.seed)
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new empty run directory')
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(device)
    spec = preprocessing(cfg, model)
    train_loader, val_loader, classes, split = training_loaders(cfg, data_root, spec, download)
    save_json(out/'config.json', cfg.to_dict()); save_json(out/'split.json', split)
    save_json(out/'preprocessing.json', spec)
    import timm, torchvision
    environment = {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'timm': timm.__version__, 'device': str(device),
                   'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   'parameters': sum(p.numel() for p in model.parameters())}
    save_json(out/'environment.json', environment)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg), weight_decay=cfg.weight_decay)
    def schedule(epoch):
        warmup = min(cfg.warmup_epochs, max(0, cfg.epochs - 1))
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup - 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    mix_fn = mixing(cfg); history = []; best = -1.; stale = 0
    for epoch in range(cfg.epochs):
        start = time.perf_counter(); rates = [g['lr'] for g in optimizer.param_groups]
        loss, updates = train_epoch(model, train_loader, optimizer, scaler, cfg, device, mix_fn)
        val, _, _ = evaluate(model, val_loader, device, cfg.num_classes)
        improved = val['accuracy'] > best
        if improved: best = val['accuracy']; stale = 0
        else: stale += 1
        history.append({'epoch': epoch, 'train_loss': loss, 'val_loss': val['loss'],
                        'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'],
                        'seconds': time.perf_counter()-start, 'learning_rates': rates,
                        'update_windows': updates})
        scheduler.step()
        checkpoint = {'format_version': 1, 'config': cfg.to_dict(), 'preprocessing': spec,
                      'classes': classes, 'class_to_idx': split['class_to_idx'],
                      'state_dict': model.state_dict(), 'epoch': epoch,
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'scaler': scaler.state_dict(), 'best_validation_accuracy': best}
        for name in (['last.pt', 'best.pt'] if improved else ['last.pt']):
            temporary = out/(name+'.tmp'); torch.save(checkpoint, temporary); temporary.replace(out/name)
        save_json(out/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        if stale >= cfg.patience: break
    plot_history(history, out/'learning_curves.png')
    print(f'Best validation accuracy: {best:.2f}%. Final test has NOT been run.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run(Config.load(args.config), args.data_root, args.output, torch.device(args.device), args.download)
