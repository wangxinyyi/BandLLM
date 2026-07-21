import argparse
import csv
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from crystallm import infer_target_layout_from_manifest, read_manifest


mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 320,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


GROUPS = ("Band",)
GROUP_COLORS = {
    "Band": "#26856c",
}
GRID_COLOR = "#d8d8d8"
TEXT_COLOR = "#222222"
FIT_COLOR = "#b04a5a"


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def masked_mae(pred, target, mask):
    mask_sum = float(mask.sum())
    if mask_sum <= 0.0:
        return float("nan")
    return float((np.abs(pred - target) * mask).sum() / mask_sum)


def masked_rmse(pred, target, mask):
    mask_sum = float(mask.sum())
    if mask_sum <= 0.0:
        return float("nan")
    return float(np.sqrt((((pred - target) ** 2) * mask).sum() / mask_sum))


def smooth_hist_line(counts, edges):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return np.empty(0), np.empty(0)
    kernel_x = np.linspace(-2.5, 2.5, 13)
    kernel = np.exp(-0.5 * kernel_x * kernel_x)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, smooth


def style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def finite_percentile_limits(values, percentile, lower_zero=False, pad_fraction=0.04):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0) if lower_zero else (-1.0, 1.0)
    if percentile and 0.0 < percentile < 100.0:
        hi = float(np.percentile(values, percentile))
        lo = float(np.percentile(values, 100.0 - percentile)) if not lower_zero else 0.0
    else:
        lo = 0.0 if lower_zero else float(values.min())
        hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0) if lower_zero else (-1.0, 1.0)
    if lo == hi:
        pad = max(abs(hi) * 0.05, 1.0)
        return (max(0.0, lo - pad), hi + pad) if lower_zero else (lo - pad, hi + pad)
    pad = (hi - lo) * pad_fraction
    return (max(0.0, lo - pad), hi + pad) if lower_zero else (lo - pad, hi + pad)


def point_density(x, y, xlim, ylim, bins=70):
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    return hist[xi, yi]


def pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def safe_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def load_predictions(manifest, pred_dir, split):
    rows = read_manifest(manifest)
    if split != "all":
        rows = [row for row in rows if row.get("split", "") == split]
    if not rows:
        raise ValueError(f"no manifest rows for split={split}")

    band_shape, _ = infer_target_layout_from_manifest(manifest, rows=rows)
    if band_shape is None:
        raise ValueError("could not infer band layout from manifest")
    band_dim = int(np.prod(band_shape))

    band_abs_sum = np.zeros(band_dim, dtype=np.float64)
    band_sq_sum = np.zeros(band_dim, dtype=np.float64)
    band_count = np.zeros(band_dim, dtype=np.float64)

    sample_rows = []
    missing = []

    for row in rows:
        sample_id = row.get("id", "")
        pred_path = os.path.join(pred_dir, f"{sample_id}.npz")
        if not os.path.exists(pred_path):
            missing.append(sample_id)
            continue

        with np.load(pred_path) as data:
            pred = data["prediction"].astype(np.float32)
            target = data["target"].astype(np.float32)
            mask = data["target_mask"].astype(np.float32)

        pred_band = pred[:band_dim]
        target_band = target[:band_dim]
        mask_band = mask[:band_dim]

        band_abs = np.abs(pred_band - target_band) * mask_band
        band_sq = ((pred_band - target_band) ** 2) * mask_band

        band_abs_sum += band_abs
        band_sq_sum += band_sq
        band_count += mask_band

        sample_rows.append(
            {
                "id": sample_id,
                "band_mae": masked_mae(pred_band, target_band, mask_band),
                "band_rmse": masked_rmse(pred_band, target_band, mask_band),
            }
        )

    if not sample_rows:
        raise ValueError(f"no prediction files found in {pred_dir}")

    with np.errstate(divide="ignore", invalid="ignore"):
        band_mae_grid = np.divide(band_abs_sum, band_count, out=np.full_like(band_abs_sum, np.nan), where=band_count > 0)
        band_rmse_grid = np.sqrt(
            np.divide(band_sq_sum, band_count, out=np.full_like(band_sq_sum, np.nan), where=band_count > 0)
        )

    return {
        "rows_total": len(rows),
        "rows_used": len(sample_rows),
        "missing": missing,
        "band_shape": tuple(band_shape),
        "sample_rows": sample_rows,
        "band_mae_grid": band_mae_grid.reshape(band_shape),
        "band_rmse_grid": band_rmse_grid.reshape(band_shape),
    }


def write_sample_csv(path, rows):
    fieldnames = ["id", "band_mae", "band_rmse"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, data):
    sample_rows = data["sample_rows"]
    metrics = {
        "band_mae": [row["band_mae"] for row in sample_rows],
        "band_rmse": [row["band_rmse"] for row in sample_rows],
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "median", "p90", "p95", "max"])
        writer.writeheader()
        for name, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            writer.writerow(
                {
                    "metric": name,
                    "mean": float(arr.mean()) if arr.size else float("nan"),
                    "median": float(np.median(arr)) if arr.size else float("nan"),
                    "p90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
                    "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
                    "max": float(arr.max()) if arr.size else float("nan"),
                }
            )


def plot_error_distributions(data, out_path, percentile):
    sample_rows = data["sample_rows"]
    values_by_group = {
        "Band": np.asarray([row["band_mae"] for row in sample_rows], dtype=np.float64),
    }
    fig, axes = plt.subplots(1, 1, figsize=(5.2, 3.9), constrained_layout=True)
    axes = [axes]
    for ax, group in zip(axes, GROUPS):
        values = values_by_group[group]
        values = values[np.isfinite(values)]
        color = GROUP_COLORS[group]
        xlim = finite_percentile_limits(values, percentile=percentile, lower_zero=True)
        plot_values = values[(values >= xlim[0]) & (values <= xlim[1])]
        if plot_values.size == 0:
            plot_values = values
        counts, edges = np.histogram(plot_values, bins=42, range=xlim, density=True)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=color,
            alpha=0.36,
            edgecolor=color,
            linewidth=0.35,
        )
        centers, smooth = smooth_hist_line(counts, edges)
        ax.plot(centers, smooth, color=color, linewidth=1.35)
        mean = safe_mean(values)
        median = safe_median(values)
        ax.axvline(mean, color=FIT_COLOR, linestyle="--", linewidth=1.1, label="Mean")
        ax.axvline(median, color=TEXT_COLOR, linestyle=":", linewidth=1.1, label="Median")
        ax.set_xlim(xlim)
        ax.set_xlabel("Sample MAE")
        ax.set_ylabel("Density")
        ax.set_title(group, fontweight="bold")
        style_axes(ax)
        ax.text(
            0.96,
            0.88,
            f"n={values.size}\nmean={mean:.3f}\nmedian={median:.3f}\np90={np.percentile(values, 90):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.legend(loc="upper center", frameon=False, ncol=2)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_band_heatmap(data, out_path, percentile):
    heatmap = data["band_mae_grid"].astype(np.float64)
    vmax_values = heatmap[np.isfinite(heatmap)]
    vmax = float(np.percentile(vmax_values, percentile)) if vmax_values.size else 1.0
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title("Band Error Heatmap", fontweight="bold")
    ax.set_xlabel("k-point index")
    ax.set_ylabel("Band index")
    ax.set_xticks(np.linspace(0, heatmap.shape[0] - 1, 6, dtype=int))
    ax.set_yticks(np.arange(0, heatmap.shape[1], max(1, heatmap.shape[1] // 8)))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Mean absolute error (eV)")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_scripts(out_dir):
    script_names = [
        "plot_fitting_curves.py",
        "plot_error_analysis_suite.py",
    ]
    bin_dir = os.path.join(PROJECT_ROOT, "bin")
    copied = []
    for name in script_names:
        src = os.path.join(bin_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main():
    parser = argparse.ArgumentParser(description="Plot a four-figure error analysis suite for electronic predictions.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--pred-dir", required=True, help="Prediction .npz directory.")
    parser.add_argument("--out-dir", required=True, help="Output folder for figures, CSVs, and plotting scripts.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--axis-percentile", type=float, default=99.5, help="Display percentile for robust axes.")
    parser.add_argument("--heatmap-percentile", type=float, default=99.0, help="Colorbar percentile for band heatmap.")
    parser.add_argument("--no-copy-scripts", action="store_true", help="Do not copy plotting scripts into out-dir.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_predictions(args.manifest, args.pred_dir, args.split)

    write_sample_csv(os.path.join(args.out_dir, "sample_error_metrics.csv"), data["sample_rows"])
    write_summary_csv(os.path.join(args.out_dir, "error_summary.csv"), data)
    np.savez_compressed(
        os.path.join(args.out_dir, "pointwise_error_arrays.npz"),
        band_mae_grid=data["band_mae_grid"].astype(np.float32),
        band_rmse_grid=data["band_rmse_grid"].astype(np.float32),
    )

    plot_error_distributions(data, os.path.join(args.out_dir, "01_error_distributions.png"), args.axis_percentile)
    plot_band_heatmap(data, os.path.join(args.out_dir, "02_band_error_heatmap.png"), args.heatmap_percentile)

    copied = [] if args.no_copy_scripts else copy_scripts(args.out_dir)

    print(f"split: {args.split}")
    print(f"manifest rows: {data['rows_total']}")
    print(f"predictions used: {data['rows_used']}")
    print(f"missing predictions: {len(data['missing'])}")
    print(f"band_shape: {data['band_shape']}")
    print(f"wrote folder: {args.out_dir}")
    print("figures:")
    for name in [
        "01_error_distributions.png",
        "02_band_error_heatmap.png",
    ]:
        print(f"  {os.path.join(args.out_dir, name)}")
    if copied:
        print("copied scripts:")
        for path in copied:
            print(f"  {path}")


if __name__ == "__main__":
    main()
