import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from tqdm import tqdm

from crystallm import DEFAULT_TARGET_BAND_COUNT, read_manifest, resolve_manifest_path, target_from_components


SPECTRUM_KEYS = {"band_energy_grid", "band_spectrum", "band_spectrum_mask"}


def main():
    parser = argparse.ArgumentParser(description="Materialize band-only 16-band line targets in existing electronic .npz samples.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite samples even when band16 exists.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows. 0 means all rows.")
    parser.add_argument("--band-count", type=int, default=DEFAULT_TARGET_BAND_COUNT)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest is empty")

    rewritten = 0
    skipped = 0
    for row in tqdm(rows, desc="materializing 16-band targets"):
        path = resolve_manifest_path(args.manifest, row["sample_path"])
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            if "band16" in files and "band16_mask" in files and "target" in files and not args.overwrite:
                skipped += 1
                continue
            arrays = {name: data[name] for name in data.files if name not in SPECTRUM_KEYS}

        band16, band16_mask, band16_indices, target, target_mask = target_from_components(
            arrays["bands"].astype(np.float32),
            arrays["band_mask"].astype(np.float32),
            arrays["dos"].astype(np.float32) if "dos" in arrays else np.zeros(0, dtype=np.float32),
            arrays["dos_mask"].astype(np.float32) if "dos_mask" in arrays else np.zeros(0, dtype=np.float32),
            target_band_count=args.band_count,
        )
        arrays["band16"] = band16
        arrays["band16_mask"] = band16_mask
        arrays["band16_indices"] = band16_indices
        arrays["target"] = target
        arrays["target_mask"] = target_mask

        tmp_path = path + ".tmp.npz"
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
        rewritten += 1

    print(f"rewritten: {rewritten}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
