import argparse
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from tqdm import tqdm

from crystallm import (
    ElectronicGridConfig,
    load_electronic_structure,
    read_cif,
    read_manifest,
    tokenize_cif_to_block,
    vectorize_electronic_structure,
    write_manifest,
)


def resolve_path(root: str, path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(root, path)


def optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Build CIF-to-electronic-structure samples.")
    parser.add_argument("manifest", help="CSV with id,cif_path and optional vasprun_path,eigenval_path,doscar_path,efermi.")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npz samples and manifest.csv.")
    parser.add_argument("--band-kpoints", type=int, default=128)
    parser.add_argument("--band-count", type=int, default=64)
    parser.add_argument("--band-axis", choices=["distance", "index"], default="distance", help="Use normalized reciprocal-space path distance or raw point index for band interpolation.")
    parser.add_argument("--band-selection", choices=["fermi", "lowest"], default="fermi", help="Select bands closest to the Fermi level or the lowest bands.")
    parser.add_argument("--band-emin", type=optional_float, default=-10.0, help="Minimum E-E_F used for the band target mask. Use 'none' to disable.")
    parser.add_argument("--band-emax", type=optional_float, default=10.0, help="Maximum E-E_F used for the band target mask. Use 'none' to disable.")
    parser.add_argument("--dos-points", type=int, default=1024)
    parser.add_argument("--dos-emin", type=float, default=-10.0)
    parser.add_argument("--dos-emax", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    config = ElectronicGridConfig(
        band_kpoints=args.band_kpoints,
        band_count=args.band_count,
        band_axis=args.band_axis,
        band_selection=args.band_selection,
        band_emin=args.band_emin,
        band_emax=args.band_emax,
        dos_points=args.dos_points,
        dos_emin=args.dos_emin,
        dos_emax=args.dos_emax,
        block_size=args.block_size,
    )

    manifest_root = os.path.dirname(os.path.abspath(args.manifest))
    rows = read_manifest(args.manifest)
    out_rows: List[Dict[str, str]] = []

    for row in tqdm(rows, desc="building electronic samples"):
        try:
            sample_id = row.get("id") or str(len(out_rows))
            cif_path = resolve_path(manifest_root, row["cif_path"])
            efermi = float(row.get("efermi") or 0.0)

            cif = read_cif(cif_path)
            input_ids, attention_mask = tokenize_cif_to_block(cif, config.block_size)
            electronic = load_electronic_structure(
                vasprun_path=resolve_path(manifest_root, row.get("vasprun_path", "")),
                eigenval_path=resolve_path(manifest_root, row.get("eigenval_path", "")),
                doscar_path=resolve_path(manifest_root, row.get("doscar_path", "")),
                efermi=efermi,
            )
            arrays = vectorize_electronic_structure(electronic, config)

            sample_name = f"{sample_id}.npz".replace(os.sep, "_")
            sample_path = os.path.join(sample_dir, sample_name)
            np.savez_compressed(
                sample_path,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **arrays,
            )
            out_rows.append({
                "id": sample_id,
                "sample_path": os.path.relpath(sample_path, args.out_dir),
                "cif_path": cif_path,
                "band_path": electronic.band_path or row.get("band_path", "unknown"),
                "source": electronic.source,
            })
        except Exception as exc:
            if not args.skip_errors:
                raise
            print(f"skipping {row.get('id', '<unknown>')}: {exc}")

    write_manifest(out_rows, os.path.join(args.out_dir, "manifest.csv"))
    print(f"wrote {len(out_rows)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
