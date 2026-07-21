import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crystallm import ElectronicStructureDataset, read_manifest, transform_target_for_training


def main():
    parser = argparse.ArgumentParser(description="Compute per-dimension target mean/std for electronic-structure samples.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--out", required=True, help="Output stats .npz path.")
    parser.add_argument("--std-floor", type=float, default=1e-6)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")

    dataset = ElectronicStructureDataset(args.manifest, rows=rows)
    sum_x = None
    sum_x2 = None
    count = None
    band_shape = None
    dos_dim = None

    for sample in dataset:
        if band_shape is None:
            band_shape = tuple(sample["bands"].shape)
            dos_dim = 0
        target = sample["target"].numpy().astype(np.float32)
        target = transform_target_for_training(target, band_shape, dos_dim).astype(np.float64)
        mask = sample["target_mask"].numpy().astype(np.float64)
        if sum_x is None:
            sum_x = np.zeros_like(target, dtype=np.float64)
            sum_x2 = np.zeros_like(target, dtype=np.float64)
            count = np.zeros_like(target, dtype=np.float64)
        sum_x += target * mask
        sum_x2 += target * target * mask
        count += mask

    safe_count = np.maximum(count, 1.0)
    mean = sum_x / safe_count
    var = np.maximum(sum_x2 / safe_count - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(count > 1, np.maximum(std, args.std_floor), 1.0)
    mean = np.where(count > 0, mean, 0.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=count.astype(np.float32),
        band_shape=np.asarray(band_shape if band_shape is not None else (), dtype=np.int32),
        dos_dim=np.asarray(-1 if dos_dim is None else dos_dim, dtype=np.int32),
        dos_transform=np.asarray("none"),
    )
    print(f"wrote stats to {args.out}")
    print(f"target_dim: {mean.shape[0]}")
    print(f"observed dims: {int((count > 0).sum())}/{count.shape[0]}")
    print(f"std mean/min/max: {float(std.mean()):.6f} / {float(std.min()):.6f} / {float(std.max()):.6f}")


if __name__ == "__main__":
    main()
