# EPOCH 42 — AI-Based Restoration of Degraded Images

## Overview

EPOCH 42 restores degraded low-resolution grayscale images (`.npy`) to the corresponding higher-resolution restored images.

The final solution uses a NAFNet-style encoder-decoder restoration network with PixelShuffle upsampling and horizontal-flip test-time augmentation (TTA).

The trained configuration uses a **2× scale factor**.

## Repository Structure

```text
EPOCH-42/
├── run.py
├── requirements.txt
├── README.md
└── models/
    ├── model.py
    ├── best_model.pth
    └── config.json
```

## Environment

The validated development/inference environment used:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- NVIDIA GPU recommended
- CUDA-capable PyTorch installation for GPU execution

The solution does not require internet access, API keys, external model downloads, or user interaction during inference.

## Installation

Install the dependencies from `requirements.txt` in an environment that already has access to the required PyTorch wheel if PyTorch is not available locally.

```bash
pip install -r requirements.txt
```

For offline evaluation, make sure the required wheels/packages are available in the evaluation environment before installation.

## Inference

The evaluator-facing entry point is `run.py`.

Run:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py TestNoisyLR TestRestored
```

### Input

The input directory must contain `.npy` files representing grayscale degraded images.

The solution accepts:

- `(H, W)`
- `(H, W, 1)`
- `(1, H, W)`

The input filename is preserved in the output.

### Output

One restored `.npy` file is generated for every input `.npy` file.

For each input:

```text
input/000000.npy
```

the corresponding output is:

```text
output/000000.npy
```

Output arrays are:

- grayscale
- shape `(H, W)`
- `float32`
- values constrained to `[0, 1]`
- checked for NaN/Inf
- restored to the target resolution determined from the input size and the configured scale factor

## Inference Details

The final configuration uses:

```text
Scale factor: 2×
Batch size: 8
```

Images are grouped by input shape before batching so tensors with different spatial dimensions are never incorrectly stacked together.

NoisyLR values are **not divided by 255**. The dataset specification allows NoisyLR values to extend slightly outside `[0, 1]`; therefore the input is passed to the trained model in its original numeric scale.

The inference pipeline uses horizontal-flip TTA:

1. Normal forward pass
2. Horizontally flipped forward pass
3. Flip the second prediction back
4. Average both predictions
5. Clamp the final prediction to `[0, 1]`
6. Save the restored `.npy`

## Model

The model is `HighResSemiconductorNet`, a NAFNet-style encoder-decoder architecture with:

- single-channel input/output
- three encoder/decoder stages
- NAF-style restoration blocks
- bottleneck blocks
- PixelShuffle upsampling
- 2× restoration output

The exact trained weights are included in:

```text
models/best_model.pth
```

The corresponding configuration is:

```text
models/config.json
```

## Validated Results

Validation results from the final trained model:

| Metric | Bicubic | EPOCH 42 |
|---|---:|---:|
| PSNR ↑ | 22.9770 dB | **29.1324 dB** |
| SSIM ↑ | 0.5243 | **0.7923** |
| LPIPS ↓ | 0.4519 | **0.2521** |

The model therefore outperformed the bicubic baseline on all three reported validation metrics.

## Test Set Verification

The standalone inference pipeline was tested on the 400-image `TestNoisyLR` set.

Verified:

- 400 / 400 input files found
- 400 / 400 images restored
- 400 restored `.npy` files generated
- corresponding filenames preserved
- output resolution follows the 2× configuration
- outputs constrained to `[0, 1]`
- no NaN/Inf outputs detected during validation

## Runtime

Measured validation runtime:

```text
GPU: NVIDIA Tesla T4
Images timed: 50
Batch size: 8
Total: 1.67 s
Average: 33.4 ms/image
```

This runtime is hardware-dependent. The official evaluation hardware may differ.

## Reproducibility

The submitted inference solution is self-contained with the trained checkpoint and configuration under `models/`.

The evaluator-facing command is only:

```bash
python run.py <input-dir> <output-dir>
```

No source-code edits or manual path configuration are required.

## External Resources

LPIPS was used as a validation metric. The trained inference model itself does not require downloading an external pretrained model or accessing an external API during evaluation.
