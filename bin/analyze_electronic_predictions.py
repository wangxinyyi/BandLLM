import argparse
import csv
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crystallm import infer_target_layout_from_manifest, read_manifest


def masked_mae(pred, target, mask):
    diff = np.abs(pred - target) * mask
    return float(diff.sum() / max(mask.sum(), 1.0))


def masked_mse(pred, target, mask):
    diff = ((pred - target) ** 2) * mask
    return float(diff.sum() / max(mask.sum(), 1.0))


def masked_rmse(pred, target, mask):
    return float(np.sqrt(max(masked_mse(pred, target, mask), 0.0)))


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:180] or "unknown"


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_value(item, key: str):
    value = item.get(key, float("nan"))
    return float(value) if np.isfinite(value) else float("inf")


def group_metric_value(item, key: str):
    value = item.get(key, float("nan"))
    return float(value) if np.isfinite(value) else -float("inf")


def select_plot_items(items, max_per_group: int, sort_metric: str):
    items_sorted = sorted(items, key=lambda item: (metric_value(item, sort_metric), item["band_mae"], item["id"]))
    if max_per_group <= 0 or len(items_sorted) <= max_per_group:
        return items_sorted
    ranks = np.linspace(0, len(items_sorted) - 1, max_per_group, dtype=np.int64)
    selected = []
    seen = set()
    for rank in ranks:
        item = items_sorted[int(rank)]
        if item["id"] in seen:
            continue
        selected.append(item)
        seen.add(item["id"])
    return selected


def prediction_metrics(pred, target, mask, band_shape, dos_dim=0):
    metrics = {
        "band_mse": float("nan"),
        "band_rmse": float("nan"),
        "band_mae": float("nan"),
    }
    if band_shape is None:
        return metrics

    band_dim = int(np.prod(band_shape))
    pred_band = pred[:band_dim]
    target_band = target[:band_dim]
    mask_band = mask[:band_dim]
    metrics.update(
        {
            "band_mse": masked_mse(pred_band, target_band, mask_band),
            "band_rmse": masked_rmse(pred_band, target_band, mask_band),
            "band_mae": masked_mae(pred_band, target_band, mask_band),
        }
    )
    return metrics


def mean_metric(items, key: str):
    values = [float(item[key]) for item in items if np.isfinite(float(item.get(key, float("nan"))))]
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze electronic predictions by band_path, export plots, and report best/worst samples."
    )
    parser.add_argument("manifest", help="Prepared manifest.csv, optionally with split column.")
    parser.add_argument("--pred-dir", required=True, help="Prediction directory written by predict_structure_to_electronic_latent.py.")
    parser.add_argument("--out-dir", required=True, help="Directory for grouped plots and summary CSV files.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--max-plots", type=int, default=0, help="Optional cap on total plots. 0 means plot every available sample.")
    parser.add_argument("--skip-plots", action="store_true", help="Only write CSV summaries; do not render plot PNGs.")
    parser.add_argument(
        "--max-plots-per-group",
        type=int,
        default=10,
        help="Maximum plots for each band_path group. 0 means plot every sample in each group.",
    )
    parser.add_argument("--top-k-groups", type=int, default=20, help="Number of path groups to print in the console summary.")
    parser.add_argument(
        "--sort-metric",
        choices=["band_mae", "band_mse"],
        default="band_mae",
        help="Metric used to rank samples and choose representative plots.",
    )
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to call plot_electronic_prediction.py.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "") == args.split]
    if not rows:
        raise ValueError(f"no manifest rows for split={args.split}")
    band_shape, dos_dim = infer_target_layout_from_manifest(args.manifest, rows=rows)

    out_dir = os.path.abspath(args.out_dir)
    plot_script = os.path.join(PROJECT_ROOT, "bin", "plot_electronic_prediction.py")
    plots_root = os.path.join(out_dir, "plots_by_band_path")
    os.makedirs(plots_root, exist_ok=True)

    sample_rows = []
    grouped = defaultdict(list)
    missing = []
    for row in rows:
        sample_id = row.get("id", "")
        pred_path = os.path.join(args.pred_dir, f"{sample_id}.npz")
        if not os.path.exists(pred_path):
            missing.append(sample_id)
            continue
        with np.load(pred_path) as pred_file:
            pred = pred_file["prediction"]
            target = pred_file["target"]
            mask = pred_file["target_mask"]
        band_path = row.get("band_path", "") or "unknown"
        metrics = prediction_metrics(pred, target, mask, band_shape, dos_dim)
        sample_info = {
            "id": sample_id,
            "band_path": band_path,
            **metrics,
        }
        sample_rows.append(sample_info)
        grouped[band_path].append(sample_info)

    if not sample_rows:
        raise ValueError("no prediction files found for the requested split")

    sample_rows.sort(key=lambda item: (metric_value(item, args.sort_metric), item["band_mae"], item["id"]))
    best_sample = sample_rows[0]
    worst_sample = max(sample_rows, key=lambda item: (metric_value(item, args.sort_metric), item["band_mae"], item["id"]))

    group_rows = []
    for band_path, items in grouped.items():
        items_sorted = sorted(items, key=lambda item: (metric_value(item, args.sort_metric), item["band_mae"], item["id"]))
        group_rows.append(
            {
                "band_path": band_path,
                "count": len(items_sorted),
                "band_mse_mean": mean_metric(items_sorted, "band_mse"),
                "band_rmse_mean": mean_metric(items_sorted, "band_rmse"),
                "band_mae_mean": mean_metric(items_sorted, "band_mae"),
                "best_id": items_sorted[0]["id"],
                "best_metric": items_sorted[0][args.sort_metric],
                "worst_id": items_sorted[-1]["id"],
                "worst_metric": items_sorted[-1][args.sort_metric],
            }
        )
    group_metric_key = f"{args.sort_metric}_mean"
    group_rows.sort(key=lambda item: (-group_metric_value(item, group_metric_key), -item["count"], item["band_path"]))

    write_csv(
        sample_rows,
        os.path.join(out_dir, "sample_metrics.csv"),
        fieldnames=[
            "id",
            "band_path",
            "band_mse",
            "band_rmse",
            "band_mae",
        ],
    )
    write_csv(
        group_rows,
        os.path.join(out_dir, "band_path_group_metrics.csv"),
        fieldnames=[
            "band_path",
            "count",
            "band_mse_mean",
            "band_rmse_mean",
            "band_mae_mean",
            "best_id",
            "best_metric",
            "worst_id",
            "worst_metric",
        ],
    )

    plots_written = 0
    if not args.skip_plots:
        for band_path, items in grouped.items():
            safe_dir = sanitize_name(band_path)
            band_path_plot_dir = os.path.join(plots_root, safe_dir)
            os.makedirs(band_path_plot_dir, exist_ok=True)
            for item in select_plot_items(items, args.max_plots_per_group, args.sort_metric):
                if args.max_plots > 0 and plots_written >= args.max_plots:
                    break
                out_png = os.path.join(band_path_plot_dir, f"{item['id']}.png")
                subprocess.run(
                    [
                        args.python_exe,
                        plot_script,
                        args.manifest,
                        "--pred-dir",
                        args.pred_dir,
                        "--id",
                        item["id"],
                        "--out",
                        out_png,
                    ],
                    check=True,
                )
                plots_written += 1
            if args.max_plots > 0 and plots_written >= args.max_plots:
                break

    write_csv(
        [
            {
                "split": args.split,
                "rows": len(rows),
                "missing_predictions": len(missing),
                "groups": len(grouped),
                "sort_metric": args.sort_metric,
                "best_id": best_sample["id"],
                "best_band_path": best_sample["band_path"],
                "best_metric": best_sample[args.sort_metric],
                "best_band_mae": best_sample["band_mae"],
                "worst_id": worst_sample["id"],
                "worst_band_path": worst_sample["band_path"],
                "worst_metric": worst_sample[args.sort_metric],
                "worst_band_mae": worst_sample["band_mae"],
                "plots_written": plots_written,
                "skip_plots": int(args.skip_plots),
                "max_plots_per_group": args.max_plots_per_group,
            }
        ],
        os.path.join(out_dir, "summary.csv"),
        fieldnames=[
            "split",
            "rows",
            "missing_predictions",
            "groups",
            "sort_metric",
            "best_id",
            "best_band_path",
            "best_metric",
            "best_band_mae",
            "worst_id",
            "worst_band_path",
            "worst_metric",
            "worst_band_mae",
            "plots_written",
            "skip_plots",
            "max_plots_per_group",
        ],
    )

    print(f"split: {args.split}")
    print(f"rows: {len(rows)}")
    print(f"missing predictions: {len(missing)}")
    print(f"band_path groups: {len(grouped)}")
    print(f"max plots per group: {args.max_plots_per_group}")
    print(f"sort metric: {args.sort_metric}")
    print(
        f"best sample: {best_sample['id']} | {args.sort_metric}={best_sample[args.sort_metric]:.6f} "
        f"| band_mae={best_sample['band_mae']:.6f} "
        f"| band_path={best_sample['band_path']}"
    )
    print(
        f"worst sample: {worst_sample['id']} | {args.sort_metric}={worst_sample[args.sort_metric]:.6f} "
        f"| band_mae={worst_sample['band_mae']:.6f} "
        f"| band_path={worst_sample['band_path']}"
    )
    print(f"worst groups by {args.sort_metric}:")
    for row in group_rows[:args.top_k_groups]:
        print(
            f"{row['count']}\t{row[group_metric_key]:.6f}\tband_mae={row['band_mae_mean']:.6f}\t{row['band_path']}"
        )
    print("top groups by count:")
    for row in sorted(group_rows, key=lambda item: (-item["count"], item["band_path"]))[:args.top_k_groups]:
        print(
            f"{row['count']}\tband_mae={row['band_mae_mean']:.6f}\t{row['band_path']}"
        )
    print(f"wrote plots to: {plots_root}")
    print(f"wrote sample metrics to: {os.path.join(out_dir, 'sample_metrics.csv')}")
    print(f"wrote group metrics to: {os.path.join(out_dir, 'band_path_group_metrics.csv')}")
    print(f"wrote summary to: {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
