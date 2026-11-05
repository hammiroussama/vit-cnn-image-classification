# CIFAR-10 benchmark results

Three selected checkpoints were evaluated on the official 10,000-image test set. All runs used the same stratified 45,000/5,000 training/validation split (seed 42); the author verified equality of saved indices and class mappings. Checkpoints were selected by validation accuracy before final testing.

| Model | Approach | Input | Epochs completed | Best epoch (1-based) | Validation accuracy | Test accuracy | Test macro F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Modified ResNet-18 | From scratch | 32 x 32 | 50 | 46 | 94.90% | 93.93% | 0.9393 |
| Custom ViT | From scratch | 32 x 32 | 50 | 46 | 82.56% | 82.08% | 0.8196 |
| ViT-Small | Pretrained, fine-tuned | 224 x 224 | 44 | 29 | 99.20% | 98.70% | 0.9870 |

The pretrained model identifier is `vit_small_patch16_224.augreg_in21k_ft_in1k`. Transfer learning uses external pretraining and larger inputs, so this is not an architecture-only comparison. Each configuration has one training seed; uncertainty across training seeds has not been measured. Imagenette results are pending.

## Interpretation

ResNet-18 exceeded the custom ViT by 11.85 percentage points on the test set under these training settings. Fine-tuned ViT-Small achieved the highest test accuracy. Test errors were 607, 1,792 and 130 respectively. These results do not establish the best possible performance of each architecture.

Validation error analysis found 52 cat/dog confusions for ResNet-18, 144 for the custom ViT and 12 for pretrained ViT-Small (both directions combined). These counts are validation findings, not test counts. Mistake galleries select the highest-scoring errors and are not representative random samples. Softmax scores are not calibrated confidence.

## Run provenance

- ResNet-18: `cifar10-cnn-scratch-20261103-201525`
- Custom ViT: `cifar10-vit-scratch-20261103-205137`
- Pretrained ViT-Small: `cifar10-vit-small-pretrained-20261105-163112`

The table was transcribed from the author's server evaluation outputs. Run `python export_portfolio.py` on that server to copy the original configuration, split, history, evaluation, environment and visualization artifacts into `benchmark_results/` and validate the reported test metrics. This command does not retrain or rerun test evaluation. Checkpoints and datasets stay outside Git.

## Demo

Run `python export_portfolio.py`, then set `MODEL_REGISTRY=demo_models.trained.json` when starting the app. The three model names state whether training used pretrained weights. Predictions on user-uploaded images may differ from benchmark performance; CIFAR-10 only provides ten supported categories.

For VS Code Remote SSH, forward server port 5000 and open the forwarded address. If local port 5000 conflicts, use local port 5001 mapped to server port 5000.
