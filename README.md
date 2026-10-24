# Vision Transformers and CNNs for Image Classification

An image classification project by **Hammir Oussama**, comparing a custom Vision Transformer with ResNet-18 and pretrained ViTs on **CIFAR-10** and **Imagenette**. The project includes training, held-out evaluation, saved experiment settings, and a local Flask prediction demo.

**Status:** the original thesis code is being consolidated into a reproducible benchmark. Corrected full-dataset results are pending server training. Historical scores from the earlier implementation are not advertised as new held-out results.

## Research questions

- How do a custom ViT and ResNet-18 compare when both are trained from scratch?
- How does transfer learning change accuracy and computational requirements?
- What mistakes remain, and can a checkpoint reproduce its evaluation and prediction preprocessing?

## Experiment design

| Track | Models | CIFAR-10 input | Imagenette input |
|---|---|---:|---:|
| Scratch | Modified ResNet-18, custom ViT | 32 × 32 | 224 × 224 |
| Transfer | ResNet-18, ViT-Small, optional ViT-Base | 224 × 224 | 224 × 224 |

Transfer learning uses resized CIFAR-10 images to preserve the pretrained input layers. Report the input-size and pretraining differences; this is not an architecture-only comparison. The custom ViT is adapted from the author's supplied implementation, with explicit dimensions and non-overlapping patches. Timm supplies pretrained ViTs; torchvision supplies ResNet-18.

**Dataset protocol:** stratified 90/10 training/development splits use a fixed seed. CIFAR-10's official test set and Imagenette's official validation set are loaded only by `evaluate.py`. Choose configurations on development results, freeze them, then run final evaluation. Earlier CIFAR runs selected checkpoints on the test set, so those scores are development evidence rather than independent test results. Imagenette is a ten-class ImageNet subset, not full ImageNet. Pretrained models may have seen these ImageNet classes during pretraining; report that overlap.

## Setup

Use Python 3.11 in a fresh environment. This compatibility baseline matches the author's installed PyTorch/timm versions; it is not a claim that these are the latest releases.

```bash
python -m venv .venv
# Linux university server
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a matching PyTorch and torchvision pair appropriate to the server driver. For example, for a compatible CUDA 12.1 setup:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU checks use `https://download.pytorch.org/whl/cpu` instead. Check the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/) and the university's environment guidance before choosing CUDA wheels. No credentials, datasets, checkpoints or personal environment folders belong in Git.

## Data

For CIFAR-10, use an existing root containing `cifar-10-batches-py`, or pass `--download` to training. For Imagenette, obtain the dataset from [fastai's official repository](https://github.com/fastai/imagenette) and supply the directory containing `train/` and `val/`, with canonical class subfolders. Data is not redistributed here.

## Train

Run all commands from the repository root. Configurations are starting points, not tuned claims. Use the same split seed across model comparisons; choose batch size based on GPU memory.

```bash
python train.py --config configs/cifar10_cnn_scratch.json --data-root /path/to/cifar-data --output runs/cifar-resnet --device cuda
python train.py --config configs/cifar10_vit.json --data-root /path/to/cifar-data --output runs/cifar-vit --device cuda
python train.py --config configs/imagenette_cnn_scratch.json --data-root /path/to/imagenette2 --output runs/imagenette-resnet --device cuda
```

Each run saves `config.json`, `split.json`, `preprocessing.json`, `environment.json`, `history.json`, learning curves, `best.pt` and `last.pt`. Existing run directories cannot be overwritten. Validation selects the best model; test evaluation is separate. Loss uses sample-weighted averaging, and gradient accumulation flushes its final partial window. MixUp/CutMix applies smoothing once; mixing drops incomplete training batches and does not report misleading hard-label training accuracy. An epoch-level warmup/cosine schedule preserves the transfer head/backbone learning-rate ratio.

Checkpoint files include optimizer, scheduler and scaler state for inspection and future recovery tooling. There is **no resume command yet**; starting training creates a new run. Seeds and split files help reproducibility, but bitwise agreement across hardware is not guaranteed.

## Final evaluation and prediction

```bash
python evaluate.py --checkpoint runs/cifar-resnet/best.pt --data-root /path/to/cifar-data --output runs/cifar-resnet/test.json --device cuda
python predict.py --checkpoint runs/cifar-resnet/best.pt --image /path/to/image.jpg
```

Evaluation records accuracy in percent, macro F1 in [0, 1], a confusion matrix and per-example labels/predictions. The checkpoint stores the architecture, ordered class labels and preprocessing. Inference loads weights strictly and never downloads a replacement pretrained model. Legacy checkpoints are explicitly rejected until migrated and verified; the old loader's silent weight filtering is not retained.

## Web demo

```bash
cp demo_models.example.json demo_models.json
# Edit checkpoint paths to point to real checkpoints produced by this version.
python app.py
```

Open `http://127.0.0.1:5000`. Add entries to the registry for the models you want to compare, and run predictions with the same image. Without a registry the demo displays a clear empty state. Images are decoded in memory, not saved to disk. The app binds to localhost with debug mode off; it is a local research demo, not a production hosting setup. Softmax scores are not calibrated confidence.

## Checks

```bash
python -m unittest discover -s tests -v
```

Offline tests cover split isolation, partial accumulation, class metrics, checkpoint strictness, prediction preprocessing, an end-to-end tiny training run, and Flask success/error paths. Synthetic checks are not benchmark results. Read [validation notes](docs/VALIDATION.md) for what was actually run.

## Portfolio milestones

1. Run short server smoke tests and record hardware and package versions.
2. Train scratch baselines, then transfer baselines. Keep ViT-Base optional.
3. Freeze configurations and evaluate held-out splits; repeat main experiments with additional seeds if feasible.
4. Add measured tables, confusion-matrix figures, failure examples and a real demo screenshot.
5. Add ONNX export and latency comparisons after correctness is established. These are planned, not implemented in this version.

## Attribution

Project and original thesis implementation: [Hammir Oussama](https://github.com/hammiroussama). Libraries and model implementations: [PyTorch](https://pytorch.org/), [torchvision](https://github.com/pytorch/vision), [timm](https://github.com/huggingface/pytorch-image-models). Dataset sources: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Imagenette](https://github.com/fastai/imagenette). Respect the applicable dataset and pretrained-weight terms. A redistribution license for the author's code has not yet been selected.
