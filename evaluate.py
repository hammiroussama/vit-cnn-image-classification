"""Run the official held-out split only after configuration selection is finished."""
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from predict import Predictor
from data import dataset
from train import evaluate
from metrics import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32); a = p.parse_args()
    if Path(a.output).exists(): raise ValueError('Refusing to overwrite an existing evaluation')
    predictor = Predictor(a.checkpoint, a.device)
    test = dataset(predictor.cfg, a.data_root, predictor.spec, 'test')
    if test.class_to_idx != predictor.class_to_idx: raise ValueError('Test label mapping differs from training')
    result, truth, predictions = evaluate(predictor.model, DataLoader(test, batch_size=a.batch_size),
                                           predictor.device, predictor.cfg.num_classes)
    result.update({'split': 'official_test' if predictor.cfg.dataset == 'cifar10' else 'official_validation_held_out',
                   'samples': len(test), 'classes': predictor.classes, 'targets': truth,
                   'predictions': predictions, 'checkpoint': str(Path(a.checkpoint).name)})
    save_json(a.output, result)
    print(f"Held-out accuracy: {result['accuracy']:.2f}%; macro F1: {result['macro_f1']:.4f}")


if __name__ == '__main__': main()
