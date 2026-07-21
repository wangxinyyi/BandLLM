import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crystallm import read_manifest, write_manifest


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def sample_band_path(root, row):
    if row.get("band_path"):
        return row["band_path"]
    sample_path = resolve(root, row["sample_path"])
    with np.load(sample_path) as data:
        if "band_path" not in data.files:
            return "unknown"
        value = data["band_path"]
        return str(value.item() if value.shape == () else value)


def absolute_row_paths(root, row):
    out = dict(row)
    if out.get("sample_path"):
        out["sample_path"] = resolve(root, out["sample_path"])
    if out.get("cif_path"):
        out["cif_path"] = resolve(root, out["cif_path"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Filter an electronic manifest by stored band-path signature.")
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--band-path", default="", help="Exact band-path signature to keep. Defaults to the most common signature.")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    root = os.path.dirname(os.path.abspath(args.manifest))
    grouped = {}
    for row in rows:
        grouped.setdefault(sample_band_path(root, row), []).append(row)

    if not grouped:
        raise ValueError("manifest is empty")

    band_path = args.band_path
    if not band_path:
        band_path = max(grouped.items(), key=lambda item: len(item[1]))[0]

    selected = [absolute_row_paths(root, row) for row in grouped.get(band_path, [])]
    if len(selected) < args.min_count:
        raise ValueError(f"band path {band_path!r} has {len(selected)} samples, below --min-count {args.min_count}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_manifest(selected, args.out)
    print(f"kept {len(selected)}/{len(rows)} samples")
    print(f"band_path: {band_path}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
