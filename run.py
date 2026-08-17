"""
KLA final evaluation entry point.

Usage:
    python run.py <input-dir> <output-dir>

Input:
    Directory containing .npy degraded grayscale images.

Output:
    One restored .npy file per input file, with the same filename.
    Output arrays are HxW, float32, values in [0, 1].
"""

import sys
import os
import json
from collections import defaultdict

import numpy as np
import torch

from models.model import HighResSemiconductorNet


BATCH_SIZE = 8


def load_array(path):
    arr = np.load(path).astype(np.float32)

    # Expected evaluation input is a grayscale HxW array.
    # Also tolerate a single-channel HxWx1 or 1xHxW array.
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]

    raise ValueError(
        f"Unsupported input shape {arr.shape} for {path}. "
        "Expected HxW, HxWx1, or 1xHxW."
    )


def main(input_dir, output_dir):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # run.py lives beside the models/ directory.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "models", "best_model.pth")
    config_path = os.path.join(root_dir, "models", "config.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model weights: {model_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing model config: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    scale_factor = int(cfg["scale_factor"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HighResSemiconductorNet(
        scale_factor=scale_factor
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    files = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".npy")
    )

    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Found {len(files)} input .npy files.")
    print(f"Device: {device}")
    print(f"Scale factor: {scale_factor}")
    print(f"Batch size: {BATCH_SIZE}")

    # Group by shape so torch.stack always receives uniform tensors.
    shape_groups = defaultdict(list)
    for path in files:
        # Only inspect shape here; full data is loaded later.
        arr = np.load(path, mmap_mode="r")
        shape_groups[arr.shape].append(path)

    with torch.no_grad():
        for shape, group_files in shape_groups.items():
            for start in range(0, len(group_files), BATCH_SIZE):
                batch_files = group_files[start:start + BATCH_SIZE]

                batch_arrays = []
                for path in batch_files:
                    arr = load_array(path)
                    batch_arrays.append(arr[None, ...])  # 1,H,W

                inp = torch.from_numpy(
                    np.stack(batch_arrays, axis=0)
                ).to(device)

                in_h, in_w = inp.shape[-2:]
                target_hw = (
                    in_h * scale_factor,
                    in_w * scale_factor,
                )

                # Same horizontal-flip TTA used by the validated standalone
                # inference pipeline.
                out1 = model(inp, target_hw=target_hw)

                flipped_inp = torch.flip(inp, dims=[3])
                out2 = model(
                    flipped_inp,
                    target_hw=target_hw
                )
                out2 = torch.flip(out2, dims=[3])

                out = (out1 + out2) / 2.0
                out = torch.nan_to_num(
                    out,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0
                )
                out = torch.clamp(out, 0.0, 1.0)

                out_np = out.cpu().numpy()

                for j, input_path in enumerate(batch_files):
                    filename = os.path.basename(input_path)
                    restored = np.asarray(
                        out_np[j, 0],
                        dtype=np.float32
                    )

                    expected_hw = target_hw
                    if restored.shape != expected_hw:
                        raise RuntimeError(
                            f"Wrong output shape for {filename}: "
                            f"got {restored.shape}, expected {expected_hw}"
                        )

                    if not np.isfinite(restored).all():
                        raise RuntimeError(
                            f"NaN/Inf detected in output {filename}"
                        )

                    restored = np.clip(
                        restored, 0.0, 1.0
                    ).astype(np.float32)

                    np.save(
                        os.path.join(output_dir, filename),
                        restored
                    )

    # Final output-count check.
    output_files = sorted(
        name for name in os.listdir(output_dir)
        if name.lower().endswith(".npy")
    )

    expected_names = sorted(
        os.path.basename(path) for path in files
    )

    if output_files != expected_names:
        missing = sorted(set(expected_names) - set(output_files))
        extra = sorted(set(output_files) - set(expected_names))
        raise RuntimeError(
            f"Output filename check failed. Missing={missing}, Extra={extra}"
        )

    print(f"Successfully restored {len(files)} / {len(files)} images.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>",
            file=sys.stderr
        )
        sys.exit(2)

    main(sys.argv[1], sys.argv[2])
