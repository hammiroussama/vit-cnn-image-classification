# First run on the university server

1. Copy this repository to your own server workspace. Keep the existing thesis project and checkpoints in their original location.
2. Check the allocated GPU with `nvidia-smi` and use your university's normal allocation procedure. This code does not request a GPU allocation or submit scheduler jobs automatically.
3. Create a fresh Python 3.11 environment and install the matching PyTorch/torchvision pair and remaining requirements from the README.
4. Run `python -m unittest discover -s tests -v` before training.
5. Copy `configs/cifar10_cnn_scratch.json` to a new configuration. For the first real server check, set epochs to 1, batch_size to 16, num_workers to 2 and warmup_epochs to 0. Leave the official test set untouched.
6. Run the command below, replacing the dataset path with your own server path. It requires the root containing `cifar-10-batches-py`.

```bash
python train.py --config configs/cifar10_server_smoke.json --data-root /path/to/cifar-data --output runs/cifar-server-smoke --device cuda
```

7. Inspect `history.json`, `environment.json` and the learning curve. If this passes, choose the full experiment budget and start a new run directory with the original full configuration. Increase batch size only if memory permits.
8. Share the GPU model, memory, environment.json and first run log for hardware-specific tuning. Do not send passwords, private keys or university login details.

Start with scratch ResNet-18 and the custom ViT. Add transfer baselines after the pipeline is stable. Fine-tuning downloads require server internet access or correctly prepopulated framework caches; a missing checkpoint will stop the run instead of silently training from scratch.

Do not compare the old checkpoint scores directly to the new runs without explaining changed splits, initialization, image size, augmentation and learning-rate schedule. Use the held-out evaluation command only after configuration selection is finished.
