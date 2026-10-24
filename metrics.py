import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def classification_metrics(labels, predictions, class_count):
    if not labels:
        raise ValueError('Cannot evaluate an empty dataset')
    return {'accuracy': 100 * float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=list(range(class_count)),
                                      average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(labels, predictions,
                                                 labels=list(range(class_count))).tolist()}


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] + 1 for r in history]
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='Train objective')
    axes[0].plot(epochs, [r['val_loss'] for r in history], label='Validation CE')
    axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[1].plot(epochs, [r['val_accuracy'] for r in history], label='Validation')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].legend()
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
