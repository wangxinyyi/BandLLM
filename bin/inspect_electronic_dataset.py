import argparse
from collections import Counter
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crystallm import read_manifest


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser(description="Inspect prepared CrystaLLM electronic-structure samples.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    root = os.path.dirname(os.path.abspath(args.manifest))
    print(f"samples: {len(rows)}")
    if not rows:
        return

    band_shapes = {}
    dos_shapes = {}
    target_dims = {}
    token_lengths = []
    band_coverages = []
    dos_coverages = []
    band_mins = []
    band_maxs = []
    band_all_mins = []
    band_all_maxs = []
    dos_mins = []
    dos_maxs = []
    dos_grid_mins = []
    dos_grid_maxs = []
    band_paths = Counter()

    for row in rows:
        path = row["sample_path"]
        sample_path = path if os.path.isabs(path) else os.path.join(root, path)
        with np.load(sample_path) as data:
            bands = data["bands"]
            band_mask = data["band_mask"]
            dos = data["dos"]
            dos_mask = data["dos_mask"]
            dos_grid = data["dos_grid"]
            target = data["target"]
            attention_mask = data["attention_mask"]
            if "band_path" in data.files:
                band_path = str(data["band_path"].item() if data["band_path"].shape == () else data["band_path"])
            else:
                band_path = row.get("band_path", "unknown")
            band_paths[band_path or "unknown"] += 1

            band_shapes[bands.shape] = band_shapes.get(bands.shape, 0) + 1
            dos_shapes[dos.shape] = dos_shapes.get(dos.shape, 0) + 1
            target_dims[target.shape[0]] = target_dims.get(target.shape[0], 0) + 1
            token_lengths.append(float(attention_mask.sum()))
            band_coverages.append(float(band_mask.mean()))
            dos_coverages.append(float(dos_mask.mean()))

            valid_bands = bands[band_mask > 0]
            if valid_bands.size:
                band_mins.append(float(valid_bands.min()))
                band_maxs.append(float(valid_bands.max()))
            if bands.size:
                band_all_mins.append(float(bands.min()))
                band_all_maxs.append(float(bands.max()))
            valid_dos = dos[dos_mask > 0]
            if valid_dos.size:
                dos_mins.append(float(valid_dos.min()))
                dos_maxs.append(float(valid_dos.max()))
            dos_grid_mins.append(float(dos_grid.min()))
            dos_grid_maxs.append(float(dos_grid.max()))

    print(f"band shapes: {band_shapes}")
    print(f"DOS shapes: {dos_shapes}")
    print(f"target dims: {target_dims}")
    print(f"CIF token length p50/p95/max: {percentile(token_lengths, 50):.0f} / {percentile(token_lengths, 95):.0f} / {max(token_lengths):.0f}")
    print(f"band mask coverage mean/min: {np.mean(band_coverages):.3f} / {np.min(band_coverages):.3f}")
    print(f"DOS mask coverage mean/min: {np.mean(dos_coverages):.3f} / {np.min(dos_coverages):.3f}")
    print(f"band energy range p01..p99 eV: {percentile(band_mins, 1):.3f} .. {percentile(band_maxs, 99):.3f}")
    print(f"band stored range p01..p99 eV: {percentile(band_all_mins, 1):.3f} .. {percentile(band_all_maxs, 99):.3f}")
    print(f"DOS value range p01..p99: {percentile(dos_mins, 1):.3f} .. {percentile(dos_maxs, 99):.3f}")
    print(f"DOS grid min/max: {min(dos_grid_mins):.3f} .. {max(dos_grid_maxs):.3f}")
    print(f"band path signatures: {len(band_paths)} unique")
    for path, count in band_paths.most_common(10):
        print(f"  {count}: {path}")

    print("examples:")
    for row in rows[:args.examples]:
        sample_path = row["sample_path"]
        path = sample_path if os.path.isabs(sample_path) else os.path.join(root, sample_path)
        with np.load(path) as data:
            print(
                f"  {row.get('id', '<unknown>')}: "
                f"tokens={int(data['attention_mask'].sum())}, "
                f"bands={data['bands'].shape}, "
                f"dos={data['dos'].shape}, "
                f"target={data['target'].shape[0]}"
            )


if __name__ == "__main__":
    main()
