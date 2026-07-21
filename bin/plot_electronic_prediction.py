import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from crystallm import read_manifest, select_target_bands


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10.5,
    "figure.dpi": 140,
    "savefig.dpi": 320,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TARGET_COLOR = "#000000"
PRED_COLOR = "#0072B2"
FERMI_COLOR = "#D55E00"
GRID_COLOR = "#E0E0E0"


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def choose_row(rows, sample_id):
    if sample_id:
        for row in rows:
            if row.get("id") == sample_id:
                return row
        raise ValueError(f"sample id not found in manifest: {sample_id}")
    return rows[0]


def format_k_label(label):
    label = label.strip()
    label = label.replace("\\Gamma", "Gamma").replace("Γ", "Gamma")
    if label in {"G", "Gamma"}:
        return r"$\Gamma$"
    match = re.fullmatch(r"([A-Za-z]+)_([0-9A-Za-z+-]+)", label)
    if match:
        return rf"${match.group(1)}_{{{match.group(2)}}}$"
    return label


def band_path_ticks(band_path, n_points):
    if not band_path or band_path == "unknown":
        return None, None
    if band_path.startswith("labels:"):
        labels = [part for part in band_path[len("labels:"):].split("-") if part]
    else:
        labels = []
        for branch in band_path.split("|"):
            parts = [part.strip().replace("\\Gamma", "G") for part in branch.split("-") if part.strip()]
            if not parts:
                continue
            if not labels or labels[-1] != parts[0]:
                labels.extend(parts)
            else:
                labels.extend(parts[1:])
    if len(labels) < 2 or n_points < 2:
        return None, None
    tick_positions = np.linspace(0.0, 1.0, len(labels), dtype=np.float32)
    tick_labels = [format_k_label(label) for label in labels]
    return tick_positions, tick_labels


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.62)
    ax.set_axisbelow(True)


def masked_stats(pred, target, mask):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    denom = max(float(mask.sum()), 1.0)
    residual = pred - target
    rmse = float(np.sqrt(((residual ** 2) * mask).sum() / denom))
    mae = float((np.abs(residual) * mask).sum() / denom)
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs target 16-band structure for one sample.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--pred-dir", required=True, help="Directory written by one of the electronic prediction scripts.")
    parser.add_argument("--id", default="", help="Material/sample id. Defaults to the first manifest row.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--print-input", action="store_true", help="Print the CIF path and first CIF lines.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")

    manifest_root = os.path.dirname(os.path.abspath(args.manifest))
    row = choose_row(rows, args.id)
    sample_id = row.get("id", "sample")
    sample_path = resolve(manifest_root, row["sample_path"])
    pred_path = os.path.join(args.pred_dir, f"{sample_id}.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    with np.load(sample_path) as sample:
        raw_bands = sample["bands"].astype(np.float32)
        raw_band_mask = sample["band_mask"].astype(np.float32)
        if "band16" in sample.files and "band16_mask" in sample.files:
            target_bands_sample = sample["band16"].astype(np.float32)
            target_band_mask_sample = sample["band16_mask"].astype(np.float32)
        else:
            target_bands_sample, target_band_mask_sample, _ = select_target_bands(raw_bands, raw_band_mask)
        band_t = sample["band_t"] if "band_t" in sample.files else np.linspace(0.0, 1.0, target_bands_sample.shape[0], dtype=np.float32)
        if "band_path" in sample.files:
            value = sample["band_path"]
            band_path = str(value.item() if getattr(value, "shape", ()) == () else value)
        else:
            band_path = row.get("band_path", "unknown")

    with np.load(pred_path) as pred_file:
        prediction = pred_file["prediction"].astype(np.float32)
        target = pred_file["target"].astype(np.float32)
        target_mask = pred_file["target_mask"].astype(np.float32)

    band_shape = target_bands_sample.shape
    band_dim = int(np.prod(band_shape))
    pred_bands = prediction[:band_dim].reshape(band_shape)
    target_bands = target[:band_dim].reshape(band_shape)
    mask_bands = target_mask[:band_dim].reshape(band_shape)
    band_rmse, band_mae = masked_stats(pred_bands, target_bands, mask_bands)

    tick_positions, tick_labels = band_path_ticks(band_path, band_shape[0])

    fig, ax_band = plt.subplots(figsize=(9.2, 5.1), constrained_layout=True)
    fig.suptitle(f"Material ID: {sample_id}", fontsize=16, fontweight="bold")

    for band_idx in range(band_shape[1]):
        valid = mask_bands[:, band_idx] > 0
        if not np.any(valid):
            continue
        ax_band.plot(
            band_t[valid],
            target_bands[valid, band_idx],
            color=TARGET_COLOR,
            linewidth=0.85,
            alpha=0.72,
            label="Target" if band_idx == 0 else None,
        )
        ax_band.plot(
            band_t[valid],
            pred_bands[valid, band_idx],
            color=PRED_COLOR,
            linewidth=1.05,
            linestyle="--",
            alpha=0.92,
            label="Prediction" if band_idx == 0 else None,
        )
    ax_band.axhline(0.0, color=FERMI_COLOR, linewidth=0.9, linestyle="--", alpha=0.82)
    if tick_positions is not None:
        ax_band.set_xticks(tick_positions)
        ax_band.set_xticklabels(tick_labels)
        for pos in tick_positions:
            ax_band.axvline(pos, color="#CFCFCF", linewidth=0.75, linestyle="--", alpha=0.72)
    style_axes(ax_band)
    ax_band.set_title(
        rf"$\mathrm{{RMSE}}_{{band}}$ = {band_rmse:.3f} eV    "
        rf"$\mathrm{{MAE}}_{{band}}$ = {band_mae:.3f} eV",
        fontsize=13.5,
        pad=9,
    )
    ax_band.set_xlabel("High-symmetry k-path")
    ax_band.set_ylabel("Energy relative to $E_F$ (eV)")
    ax_band.set_xlim(0.0, 1.0)
    ax_band.margins(x=0.0, y=0.06)
    ax_band.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", handlelength=2.8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"id: {sample_id}")
    print(f"sample: {sample_path}")
    print(f"prediction: {pred_path}")
    print(f"band RMSE/MAE (eV): {band_rmse:.6f} / {band_mae:.6f}")
    print(f"wrote plot: {args.out}")

    if args.print_input:
        cif_path = row.get("cif_path", "")
        if cif_path:
            resolved_cif = resolve(manifest_root, cif_path)
            print(f"CIF input path: {resolved_cif}")
            with open(resolved_cif, "rt", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 30:
                        break
                    print(line.rstrip())


if __name__ == "__main__":
    main()
