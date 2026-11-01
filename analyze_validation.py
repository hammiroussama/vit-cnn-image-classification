"""CPU-only analysis of the saved CIFAR-10 validation split (never the test set)."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from data import dataset
from predict import Predictor


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir, data_root, output=None, batch_size=32, threads=2):
    if batch_size < 1 or threads < 1:
        raise ValueError('Batch size and threads must be positive')
    torch.set_num_threads(threads)
    run_dir = Path(run_dir)
    predictor = Predictor(run_dir / 'best.pt', device='cpu')
    if predictor.cfg.dataset != 'cifar10':
        raise ValueError('This analysis script is specifically for CIFAR-10')
    manifest = json.loads((run_dir / 'split.json').read_text())
    if manifest.get('source_split') != 'training':
        raise ValueError('Expected a validation subset of the training data')
    images = dataset(predictor.cfg, data_root, predictor.spec, 'train', download=False)
    if not (images.class_to_idx == manifest['class_to_idx'] == predictor.class_to_idx):
        raise ValueError('Dataset, split and checkpoint class mappings differ')
    train_ids, val_ids = manifest['train_indices'], manifest['validation_indices']
    for indices in (train_ids, val_ids):
        if not indices or any(type(i) is not int or not 0 <= i < len(images) for i in indices):
            raise ValueError('Invalid or empty saved indices')
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate saved indices')
    if set(train_ids) & set(val_ids):
        raise ValueError('Training and validation indices overlap')
    if len(train_ids) + len(val_ids) != len(images):
        raise ValueError('Saved split does not cover this dataset')
    output = Path(output) if output else run_dir / ('validation_analysis_' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --output path')
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(Subset(images, val_ids), batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    print(f'CPU only: analyzing {len(val_ids)} validation images.', flush=True)
    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(loader):
            logits = predictor.model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError('Model returned non-finite outputs')
            scores, predictions = logits.softmax(1).max(1)
            for target, prediction, score in zip(targets.tolist(), predictions.tolist(), scores.tolist()):
                rows.append(dict(dataset_index=val_ids[len(rows)], true_id=target,
                                 predicted_id=prediction, true_class=predictor.classes[target],
                                 predicted_class=predictor.classes[prediction], softmax_score=score))
            if batch % 10 == 0 or len(rows) == len(val_ids):
                print(f'Processed {len(rows)}/{len(val_ids)}', flush=True)
    truth = [r['true_id'] for r in rows]
    predicted = [r['predicted_id'] for r in rows]
    labels = list(range(len(predictor.classes)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    per_class = [dict(class_name=name, support=int(support[i]), correct=int(matrix[i, i]),
                      precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                      accuracy_percent=float(recall[i]*100) if support[i] else None)
                 for i, name in enumerate(predictor.classes)]
    summary = dict(dataset='cifar10', split='saved validation subset of training data',
                   checkpoint=str(run_dir / 'best.pt'), device='cpu', samples=len(rows),
                   accuracy_percent=float(np.trace(matrix) / len(rows) * 100),
                   macro_f1=float(f1.mean()),
                   note='Exploratory validation results, not test results. Softmax scores are not calibrated confidence.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(output / 'validation_predictions.csv', rows)
    write_csv(output / 'per_class_metrics.csv', per_class)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Number of images')
    ax.set(xticks=labels, yticks=labels, xticklabels=predictor.classes, yticklabels=predictor.classes,
           xlabel='Predicted class', ylabel='True class', title='CIFAR-10: saved validation split')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in labels:
        for j in labels:
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(predictor.classes, [r['accuracy_percent'] if r['support'] else np.nan for r in per_class], color='#246c60')
    ax.set(xlim=(0, 100), xlabel='Correct predictions within each true class (%)', title='Validation accuracy by class')
    fig.tight_layout()
    fig.savefig(output / 'per_class_accuracy.png', dpi=160)
    plt.close(fig)
    errors = sorted((r for r in rows if r['true_id'] != r['predicted_id']), key=lambda r: r['softmax_score'], reverse=True)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, errors):
        ax.imshow(images.data[row['dataset_index']], interpolation='nearest')
        ax.set_title(f"True: {row['true_class']}\nPredicted: {row['predicted_class']} ({row['softmax_score']:.1%})", fontsize=10)
    fig.suptitle('Validation mistakes with highest softmax scores\nScores are not calibrated confidence' if errors else 'No validation mistakes')
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / 'mistakes.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f'Results saved to: {output.resolve()}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    analyze(**vars(args))
