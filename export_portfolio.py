"""Export existing server evidence and create a three-model demo registry."""
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

RUNS = [
    ('resnet18', 'ResNet-18 - from scratch', 'cifar10-cnn-scratch-20261103-201525', 93.93, 0.9393),
    ('vit_custom', 'Custom ViT - from scratch', 'cifar10-vit-scratch-20261103-205137', 82.08, 0.8196),
    ('vit_small', 'ViT-Small - pretrained, fine-tuned', 'cifar10-vit-small-pretrained-20261105-163112', 98.70, 0.9870),
]


def export(root):
    root = Path(root)
    prepared = []
    reference = None
    for key, label, name, accuracy, f1 in RUNS:
        run = root / 'runs' / name
        required = ['config.json', 'split.json', 'history.json', 'preprocessing.json', 'environment.json', 'test_results.json', 'best.pt']
        for filename in required:
            if not (run / filename).is_file():
                raise FileNotFoundError(run / filename)
        split = json.loads((run / 'split.json').read_text())
        signature = {k: split[k] for k in ('train_indices', 'validation_indices', 'class_to_idx')}
        if reference is not None and signature != reference:
            raise ValueError('Run splits or class mappings differ')
        reference = signature
        result = json.loads((run / 'test_results.json').read_text())
        if result.get('split') != 'official_test' or result.get('samples') != 10000:
            raise ValueError(f'{name}: expected official 10,000-image test evaluation')
        if abs(result['accuracy'] - accuracy) > 0.005 or abs(result['macro_f1'] - f1) > 0.000051:
            raise ValueError(f'{name}: saved metrics differ from reported results')
        history = json.loads((run / 'history.json').read_text())
        best = max(history, key=lambda row: row['val_accuracy'])
        prepared.append((key, label, name, run, result, history, best))
    destination = root / 'benchmark_results' / datetime.now().strftime('%Y%m%d-%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    rows, registry = [], {}
    for key, label, name, run, result, history, best in prepared:
        target = destination / key
        target.mkdir()
        for filename in ('config.json', 'split.json', 'history.json', 'preprocessing.json', 'environment.json', 'test_results.json', 'learning_curves.png'):
            if (run / filename).exists():
                shutil.copy2(run / filename, target / filename)
        for analysis in sorted(run.glob('validation_analysis_*')):
            if analysis.is_dir():
                for item in analysis.iterdir():
                    if item.is_file() and item.suffix in ('.png', '.csv', '.json'):
                        folder = target / analysis.name
                        folder.mkdir(exist_ok=True)
                        shutil.copy2(item, folder / item.name)
        rows.append(dict(model=label, run=name, epochs=len(history), best_epoch=best['epoch']+1,
                         validation_accuracy=best['val_accuracy'], test_accuracy=result['accuracy'], test_macro_f1=result['macro_f1']))
        registry[key] = dict(name=label, dataset='cifar10', checkpoint=f'runs/{name}/best.pt')
    with (destination / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    registry_path = root / 'demo_models.trained.json'
    if registry_path.exists():
        shutil.copy2(registry_path, destination / 'previous_demo_registry.json')
    registry_path.write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')
    print(f'Exported verified evidence: {destination}')
    print(f'Demo registry: {registry_path}')
    for row in rows:
        print(f"{row['model']}: test accuracy {row['test_accuracy']:.2f}%, macro F1 {row['test_macro_f1']:.4f}")
    return destination


if __name__ == '__main__':
    export(Path(__file__).resolve().parent)
