# Validation

Checked locally on 24 October 2026 using Python 3.11, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121 and timm 0.9.16. CUDA was not available on this machine, so all executed model checks used CPU.

## Passed

- Nine offline regression tests, including a tiny full train/save/load/predict cycle and valid/invalid Flask requests.
- Disjoint, repeatable stratified splits; training code does not request the official test set.
- Final partial gradient accumulation agrees numerically with a single full batch for a simple linear model.
- Missing checkpoint parameters raise an error; pretrained load failures propagate rather than changing the experiment silently.
- Real CIFAR-10 loader: 45,000 training and 5,000 development examples. Two optimizer updates on four real images completed with finite loss.
- Real Imagenette loader: 8,522 training and 947 development examples. Two optimizer updates on four real images completed with finite loss.
- Offline forward passes for ResNet-18, custom ViT, ViT-Small and ViT-Base at 224px produced finite outputs of shape [1, 10]. Pretrained downloads were disabled for these architecture checks.
- JavaScript syntax check and browser inspection of the demo's empty state. The classify button was disabled, and the page accurately stated that no trained models were registered.

## Environment detail

The source environment had Flask 2.3.2 paired with a newer Werkzeug, which breaks Flask's own test-client helper. Endpoint regression tests used Werkzeug's WSGI client directly and passed against the actual app. The new requirements request Flask 3.x. A clean installation of those requirements and CI execution have not yet been verified; no original environment packages were modified.

## Not yet verified

Full-dataset training, GPU mixed precision, pretrained weight downloads, university-server execution, final held-out accuracy, ONNX export and latency benchmarks remain pending. Synthetic and four-image smoke checks are not evidence of useful classification accuracy. Legacy checkpoint migration and numerical equivalence remain pending.
