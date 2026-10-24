# Changes from the supplied thesis project

- Unified both datasets behind explicit configurations and one trainer.
- Retained the custom ViT source structure; corrected its non-overlapping-patch description and validated head dimensions.
- Added a working CIFAR scratch ResNet path; the supplied text draft was missing a closing parenthesis and the CIFAR trainer did not expose it.
- Preserved pretrained ResNet input weights. Scratch CIFAR uses a 3×3 stride-one stem; transfer models retain their pretrained stem.
- Replaced manual partial ViT weight copying with explicit timm models. The initial corrected transfer baseline uses 224px inputs for both datasets. The old 32px partial-transfer variant is not represented as equivalent.
- Added stratified development splits, saved indices and class mappings, and independent final evaluation.
- Applied MixUp smoothing once and enabled CutMix-only configurations. Removed CUDA-specific random erasing from CPU data transforms.
- Corrected classifier parameter groups, CPU mixed-precision handling, final accumulation windows and per-sample loss averages.
- Replaced batch OneCycle scheduling with a documented epoch warmup/cosine schedule that maintains group ratios.
- Save best and latest checkpoints separately; write strict JSON without NaN training-accuracy placeholders.
- Replaced inference's hardcoded architectures, dropped convolution weights, shape filtering and inconsistent normalization with checkpoint metadata and strict loading.
- Replaced the fragile dashboard with a small dependency-free local prediction interface. The original script called undefined updateCharts and reused empty canvas identifiers; the new demo has explicit model selection, errors and empty states. Historical metric charts are not displayed as verified corrected results.
- Fixed the declared torch/torchvision mismatch to the installed matching 2.5.1/0.20.1 pair.

## Existing artifacts

The original project contains trained checkpoints and historical metrics. Originals remain in the user's source directory and are excluded from this repository. Their source settings and model shapes must be reconstructed before migration. The new code refuses them rather than silently accepting incomplete weights. Historical CIFAR best scores were selected using the official test set; rerun the corrected protocol before making held-out performance claims.
