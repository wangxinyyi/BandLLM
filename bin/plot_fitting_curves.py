import argparse
import csv
import os
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
        "figure.dpi": 140,
        "savefig.dpi": 320,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


PANEL_COLORS = {
    "Band": "#26856c",
}
GRID_COLOR = "#d8d8d8"
IDENTITY_COLOR = "#111111"
FIT_COLOR = "#b04a5a"
TEXT_BOX = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#cccccc", linewidth=0.5, alpha=0.7)


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0
        self.sae = 0.0
        self.sse = 0.0

    def update(self, actual, pred):
        actual = np.asarray(actual, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if actual.size == 0:
            return
        diff = pred - actual
        self.count += int(actual.size)
        self.sx += float(actual.sum())
        self.sy += float(pred.sum())
        self.sxx += float(np.dot(actual, actual))
        self.syy += float(np.dot(pred, pred))
        self.sxy += float(np.dot(actual, pred))
        self.sae += float(np.abs(diff).sum())
        self.sse += float(np.dot(diff, diff))

    def metrics(self):
        if self.count <= 0:
            return {
                "n": 0,
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
            }
        n = float(self.count)
        x_mean = self.sx / n
        y_mean = self.sy / n
        sst = self.sxx - self.sx * self.sx / n
        var_x = self.sxx - self.sx * self.sx / n
        cov_xy = self.sxy - self.sx * self.sy / n
        slope = cov_xy / var_x if abs(var_x) > 1e-12 else float("nan")
        intercept = y_mean - slope * x_mean if np.isfinite(slope) else float("nan")
        r2 = 1.0 - self.sse / sst if abs(sst) > 1e-12 else float("nan")
        return {
            "n": self.count,
            "r2": float(r2),
            "rmse": float(np.sqrt(max(self.sse / n, 0.0))),
            "mae": float(self.sae / n),
            "slope": float(slope),
            "intercept": float(intercept),
        }


class ReservoirSampler:
    def __init__(self, max_points, seed):
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self.actual = np.empty(0, dtype=np.float32)
        self.pred = np.empty(0, dtype=np.float32)
        self.seen = 0

    def update(self, actual, pred):
        actual = np.asarray(actual, dtype=np.float32).reshape(-1)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if actual.size == 0 or self.max_points <= 0:
            self.seen += int(actual.size)
            return

        size = int(actual.size)
        self.seen += size
        current = int(self.actual.size)
        free = max(self.max_points - current, 0)
        if free > 0:
            take = min(free, size)
            self.actual = np.concatenate([self.actual, actual[:take]])
            self.pred = np.concatenate([self.pred, pred[:take]])
            actual = actual[take:]
            pred = pred[take:]
            size = int(actual.size)
            if size == 0:
                return

        positions = self.rng.integers(0, self.seen, size=size)
        keep = positions < self.max_points
        if np.any(keep):
            replace_idx = positions[keep].astype(np.int64)
            self.actual[replace_idx] = actual[keep]
            self.pred[replace_idx] = pred[keep]


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def finite_mask(actual, pred, mask):
    valid = np.asarray(mask).reshape(-1) > 0
    actual = np.asarray(actual).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    valid &= np.isfinite(actual) & np.isfinite(pred)
    return actual[valid], pred[valid]


def load_prediction_parts(path, band_dim):
    with np.load(path) as data:
        pred = data["prediction"].astype(np.float32)
        target = data["target"].astype(np.float32)
        mask = data["target_mask"].astype(np.float32)
    pred_band = pred[:band_dim]
    target_band = target[:band_dim]
    mask_band = mask[:band_dim]
    return target_band, pred_band, mask_band


def update_group(stats, sampler, actual, pred, mask):
    actual_valid, pred_valid = finite_mask(actual, pred, mask)
    stats.update(actual_valid, pred_valid)
    sampler.update(actual_valid, pred_valid)


def axis_limits(actual, pred, percentile, pad_fraction=0.04):
    values = np.concatenate([actual, pred]).astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    if percentile and 0.0 < percentile < 100.0:
        lo, hi = np.percentile(values, [100.0 - percentile, percentile])
    else:
        lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return float(lo - pad), float(hi + pad)


def point_density(actual, pred, limits, bins=80):
    x_min, x_max = limits
    y_min, y_max = limits
    hist, x_edges, y_edges = np.histogram2d(actual, pred, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    xi = np.clip(np.searchsorted(x_edges, actual, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges, pred, side="right") - 1, 0, bins - 1)
    dens = hist[xi, yi]
    return dens


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


def format_metric(value):
    return f"{value:.3f}" if np.isfinite(value) else "nan"


def style_main_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def plot_panel(fig, outer_spec, title, actual, pred, metrics, color, percentile):
    inner = outer_spec.subgridspec(
        2,
        2,
        height_ratios=[0.28, 1.0],
        width_ratios=[1.0, 0.28],
        hspace=0.02,
        wspace=0.02,
    )
    ax_histx = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(inner[1, 1], sharey=ax_main)
    fig.add_subplot(inner[0, 1]).axis("off")

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
    order = np.argsort(density)
    ax_main.scatter(
        plot_actual[order],
        plot_pred[order],
        c=density[order],
        cmap="Spectral_r",
        s=6,
        alpha=0.62,
        edgecolors="none",
        rasterized=True,
    )
    line_x = np.asarray(limits, dtype=np.float64)
    ax_main.plot(line_x, line_x, color=IDENTITY_COLOR, linewidth=1.15)
    if np.isfinite(metrics["slope"]) and np.isfinite(metrics["intercept"]):
        ax_main.plot(
            line_x,
            metrics["slope"] * line_x + metrics["intercept"],
            color=FIT_COLOR,
            linestyle="--",
            linewidth=1.2,
        )

    ax_main.set_xlim(limits)
    ax_main.set_ylim(limits)
    ax_main.set_xlabel("Actual Value")
    ax_main.set_ylabel("Predicted Value")
    style_main_axes(ax_main)
    ax_main.text(
        0.96,
        0.08,
        "$R^2$="
        + format_metric(metrics["r2"])
        + "\nRMSE="
        + format_metric(metrics["rmse"])
        + "\nMAE="
        + format_metric(metrics["mae"]),
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        linespacing=1.12,
        bbox=TEXT_BOX,
        zorder=8,
    )

    bins = 44
    counts_x, edges_x = np.histogram(plot_actual, bins=bins, range=limits)
    ax_histx.bar(
        edges_x[:-1],
        counts_x,
        width=np.diff(edges_x),
        align="edge",
        color=color,
        alpha=0.42,
        edgecolor=color,
        linewidth=0.35,
    )
    centers_x, smooth_x = smooth_hist_line(counts_x, edges_x)
    ax_histx.plot(centers_x, smooth_x, color=color, linewidth=1.0)
    ax_histx.set_title(title, pad=4, fontweight="bold")
    ax_histx.axis("off")

    counts_y, edges_y = np.histogram(plot_pred, bins=bins, range=limits)
    ax_histy.barh(
        edges_y[:-1],
        counts_y,
        height=np.diff(edges_y),
        align="edge",
        color=color,
        alpha=0.42,
        edgecolor=color,
        linewidth=0.35,
    )
    centers_y, smooth_y = smooth_hist_line(counts_y, edges_y)
    ax_histy.plot(smooth_y, centers_y, color=color, linewidth=1.0)
    ax_histy.axis("off")
    return ax_main


def write_metrics_csv(path, metrics_by_name):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n", "r2", "rmse", "mae", "slope", "intercept"])
        writer.writeheader()
        for name, metrics in metrics_by_name.items():
            writer.writerow({"group": name, **metrics})


def main():
    parser = argparse.ArgumentParser(description="Plot actual-vs-predicted fitting curves for band targets.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--pred-dir", required=True, help="Directory with prediction .npz files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for fitting curve plots.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--max-points", type=int, default=180000, help="Reservoir-sampled scatter points per panel.")
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=99.5,
        help="Symmetric display percentile for axes. Use 100 to show full sampled range.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--combined-name", default="fitting_curves.png")
    parser.add_argument("--separate", action="store_true", help="Also write one PNG per panel.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "") == args.split]
    if not rows:
        raise ValueError(f"no manifest rows for split={args.split}")

    band_shape, _ = infer_target_layout_from_manifest(args.manifest, rows=rows)
    if band_shape is None:
        raise ValueError("could not infer band layout from manifest")
    band_dim = int(np.prod(band_shape))

    groups = ("Band",)
    stats = {name: RunningStats() for name in groups}
    samplers = {
        "Band": ReservoirSampler(args.max_points, args.seed + 17),
    }
    missing = []
    used = 0

    for row in rows:
        sample_id = row.get("id", "")
        pred_path = os.path.join(args.pred_dir, f"{sample_id}.npz")
        if not os.path.exists(pred_path):
            missing.append(sample_id)
            continue
        band = load_prediction_parts(pred_path, band_dim)
        update_group(stats["Band"], samplers["Band"], *band)
        used += 1

    if used == 0:
        raise ValueError("no prediction files found")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_by_name = {name: stats[name].metrics() for name in groups}
    write_metrics_csv(os.path.join(args.out_dir, "fitting_curve_metrics.csv"), metrics_by_name)

    fig = plt.figure(figsize=(4.9, 4.9), constrained_layout=True)
    outer = fig.add_gridspec(1, 1)
    for idx, name in enumerate(groups):
        plot_panel(
            fig,
            outer[idx],
            name,
            samplers[name].actual,
            samplers[name].pred,
            metrics_by_name[name],
            PANEL_COLORS[name],
            args.axis_percentile,
        )

    combined_path = os.path.join(args.out_dir, args.combined_name)
    fig.savefig(combined_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if args.separate and len(groups) > 1:
        for name in groups:
            sep_fig = plt.figure(figsize=(4.5, 4.6), constrained_layout=True)
            outer_single = sep_fig.add_gridspec(1, 1)
            plot_panel(
                sep_fig,
                outer_single[0],
                name,
                samplers[name].actual,
                samplers[name].pred,
                metrics_by_name[name],
                PANEL_COLORS[name],
                args.axis_percentile,
            )
            sep_path = os.path.join(args.out_dir, f"fitting_{name.lower()}.png")
            sep_fig.savefig(sep_path, bbox_inches="tight", facecolor="white")
            plt.close(sep_fig)

    print(f"split: {args.split}")
    print(f"manifest rows: {len(rows)}")
    print(f"predictions used: {used}")
    print(f"missing predictions: {len(missing)}")
    print(f"band_shape: {band_shape}")
    print(f"scatter points per panel cap: {args.max_points}")
    for name, metrics in metrics_by_name.items():
        print(
            f"{name}: n={metrics['n']} R2={metrics['r2']:.6f} "
            f"RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
            f"slope={metrics['slope']:.6f} intercept={metrics['intercept']:.6f}"
        )
    print(f"wrote plot: {combined_path}")
    if args.separate and len(groups) <= 1:
        print("separate plot skipped: only one band panel is present")
    print(f"wrote metrics: {os.path.join(args.out_dir, 'fitting_curve_metrics.csv')}")


if __name__ == "__main__":
    main()
