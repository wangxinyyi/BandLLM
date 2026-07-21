import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from crystallm import read_manifest


mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.dpi": 140,
        "savefig.dpi": 320,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BAND_COLOR = "#2f8f76"
FIT_COLOR = "#c85a5a"
IDENTITY_COLOR = "#1a1a1a"
GRID_COLOR = "#d8d8d8"
TEXT_BOX = dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", linewidth=0.65, alpha=0.92)


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def choose_row(rows, sample_id):
    for row in rows:
        if row.get("id") == sample_id:
            return row
    raise ValueError(f"sample id not found in manifest: {sample_id}")


def masked_valid(actual, pred, mask):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    valid = (mask > 0) & np.isfinite(actual) & np.isfinite(pred)
    return actual[valid], pred[valid]


def metrics(actual, pred):
    if actual.size == 0:
        return {
            "n": 0,
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }
    diff = pred - actual
    sse = float(np.dot(diff, diff))
    sae = float(np.abs(diff).sum())
    mean = float(actual.mean())
    sst = float(np.dot(actual - mean, actual - mean))
    var_x = float(np.dot(actual - mean, actual - mean))
    cov = float(np.dot(actual - mean, pred - float(pred.mean())))
    slope = cov / var_x if abs(var_x) > 1e-12 else float("nan")
    intercept = float(pred.mean()) - slope * mean if np.isfinite(slope) else float("nan")
    return {
        "n": int(actual.size),
        "r2": 1.0 - sse / sst if abs(sst) > 1e-12 else float("nan"),
        "rmse": float(np.sqrt(max(sse / float(actual.size), 0.0))),
        "mae": float(sae / float(actual.size)),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def axis_limits(actual, pred, percentile):
    values = np.concatenate([actual, pred]).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    if percentile and 0.0 < percentile < 100.0:
        lo = float(np.percentile(values, 100.0 - percentile))
        hi = float(np.percentile(values, percentile))
    else:
        lo = float(values.min())
        hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        pad = max(abs(hi) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def point_density(actual, pred, limits, bins=55):
    if actual.size == 0:
        return np.empty(0, dtype=np.float64)
    hist, x_edges, y_edges = np.histogram2d(actual, pred, bins=bins, range=[limits, limits])
    xi = np.clip(np.searchsorted(x_edges, actual, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges, pred, side="right") - 1, 0, bins - 1)
    return hist[xi, yi]


def style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def fmt(value):
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def plot_panel(ax, title, actual, pred, color, units, percentile):
    stats = metrics(actual, pred)
    limits = axis_limits(actual, pred, percentile=percentile)
    in_view = (
        (actual >= limits[0])
        & (actual <= limits[1])
        & (pred >= limits[0])
        & (pred <= limits[1])
        & np.isfinite(actual)
        & np.isfinite(pred)
    )
    plot_actual = actual[in_view]
    plot_pred = pred[in_view]
    if plot_actual.size == 0:
        plot_actual = actual
        plot_pred = pred

    density = point_density(plot_actual, plot_pred, limits)
    order = np.argsort(density) if density.size else np.arange(plot_actual.size)
    ax.scatter(
        plot_actual[order],
        plot_pred[order],
        c=density[order] if density.size else color,
        cmap="Spectral_r",
        s=14,
        alpha=0.78,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(limits, limits, color=IDENTITY_COLOR, linewidth=1.35, label="y = x")
    if np.isfinite(stats["slope"]) and np.isfinite(stats["intercept"]):
        line_x = np.asarray(limits, dtype=np.float64)
        ax.plot(
            line_x,
            stats["slope"] * line_x + stats["intercept"],
            color=FIT_COLOR,
            linewidth=1.25,
            linestyle="--",
            label="Linear fit",
        )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(f"Actual Value{units}")
    ax.set_ylabel(f"Predicted Value{units}")
    style_axes(ax)
    ax.text(
        0.96,
        0.05,
        "$R^2$="
        + fmt(stats["r2"])
        + "\nRMSE="
        + fmt(stats["rmse"])
        + "\nMAE="
        + fmt(stats["mae"])
        + f"\nn={stats['n']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=TEXT_BOX,
    )
    ax.legend(loc="upper left", frameon=False)


def main():
    parser = argparse.ArgumentParser(description="Plot per-sample band actual-vs-predicted fitting panel.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--pred-dir", required=True, help="Prediction .npz directory.")
    parser.add_argument("--id", required=True, help="Material/sample id.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--axis-percentile", type=float, default=99.5)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    row = choose_row(rows, args.id)
    manifest_root = os.path.dirname(os.path.abspath(args.manifest))
    sample_path = resolve(manifest_root, row["sample_path"])
    pred_path = os.path.join(args.pred_dir, f"{args.id}.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    with np.load(sample_path) as sample:
        if "band16" in sample.files:
            band_shape = tuple(sample["band16"].shape)
        elif "bands" in sample.files:
            band_shape = tuple(sample["bands"].shape)
        else:
            raise ValueError(f"sample has no band array: {sample_path}")

    band_dim = int(np.prod(band_shape))
    with np.load(pred_path) as data:
        pred = data["prediction"].astype(np.float32)
        target = data["target"].astype(np.float32)
        mask = data["target_mask"].astype(np.float32)

    target_band = target[:band_dim]
    pred_band = pred[:band_dim]
    mask_band = mask[:band_dim]

    band_actual, band_pred = masked_valid(target_band, pred_band, mask_band)

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.4), constrained_layout=True)
    fig.suptitle(f"Material ID: {args.id}", fontsize=17, fontweight="bold", y=1.05)
    plot_panel(ax, "Band Fitting", band_actual, band_pred, BAND_COLOR, " (eV)", args.axis_percentile)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"id: {args.id}")
    print(f"sample: {sample_path}")
    print(f"prediction: {pred_path}")
    print(f"wrote fitting plot: {args.out}")


if __name__ == "__main__":
    main()
