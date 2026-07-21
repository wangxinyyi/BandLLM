import argparse
import csv
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from crystallm import infer_target_layout_from_manifest, read_manifest


ERROR_METRIC_NAMES = ("band_mae", "band_rmse")
R2_METRIC_NAMES = ("band_r2",)
METRIC_NAMES = ERROR_METRIC_NAMES + R2_METRIC_NAMES
HIGHER_IS_BETTER = set(R2_METRIC_NAMES)


def sanitize_name(value, max_len=96):
    text = str(value or "unknown").strip() or "unknown"
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in "._-+=" else "_")
    text = "".join(keep).strip("._")
    if len(text) > max_len:
        text = text[:max_len].rstrip("._")
    return text or "unknown"


def r2_score(target, pred, mask):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    valid = (mask > 0) & np.isfinite(target) & np.isfinite(pred)
    target = target[valid]
    pred = pred[valid]
    if target.size < 2:
        return float("nan")
    diff = pred - target
    sse = float(np.dot(diff, diff))
    centered = target - float(target.mean())
    sst = float(np.dot(centered, centered))
    return 1.0 - sse / sst if abs(sst) > 1e-12 else float("nan")


def add_r2_metrics(out, data, band_dim):
    out["band_r2"] = float("nan")
    if band_dim is None:
        return
    if not {"prediction", "target", "target_mask"} <= set(data.files):
        return
    pred = data["prediction"]
    target = data["target"]
    mask = data["target_mask"]
    band_dim = int(band_dim)
    out["band_r2"] = r2_score(target[:band_dim], pred[:band_dim], mask[:band_dim])


def load_metric_row(row, pred_dir, band_dim):
    sample_id = row.get("id", "")
    pred_path = os.path.join(pred_dir, f"{sample_id}.npz")
    if not os.path.exists(pred_path):
        return None
    with np.load(pred_path) as data:
        out = {
            "id": sample_id,
            "split": row.get("split", ""),
            "band_path": row.get("band_path", "") or "unknown",
            "pred_path": pred_path,
        }
        for metric in ERROR_METRIC_NAMES:
            if metric in data.files:
                out[metric] = float(np.asarray(data[metric]).reshape(()))
            else:
                out[metric] = float("nan")
        add_r2_metrics(out, data, band_dim)
    return out


def metric_value(item, metric):
    value = float(item.get(metric, float("nan")))
    return value if np.isfinite(value) else float("inf")


def sort_key(item, metric, best_first=True):
    value = float(item.get(metric, float("nan")))
    if not np.isfinite(value):
        return (1, 0.0, item["id"])
    if metric in HIGHER_IS_BETTER:
        ordered_value = -value if best_first else value
    else:
        ordered_value = value if best_first else -value
    return (0, ordered_value, item["id"])


def take_unique(candidates, count, used_ids):
    selected = []
    for item in candidates:
        if len(selected) >= count:
            break
        if item["id"] in used_ids:
            continue
        selected.append(item)
        used_ids.add(item["id"])
    return selected


def select_quality_sets(items, metric, count):
    ordered = sorted(items, key=lambda item: sort_key(item, metric, best_first=True))
    used_ids = set()
    good = take_unique(ordered, count, used_ids)
    bad = take_unique(sorted(items, key=lambda item: sort_key(item, metric, best_first=False)), count, used_ids)

    center = (len(ordered) - 1) / 2.0
    middle_candidates = sorted(
        enumerate(ordered),
        key=lambda pair: (abs(pair[0] - center), pair[0]),
    )
    middle = take_unique((item for _, item in middle_candidates), count, used_ids)
    return {"good": good, "middle": middle, "bad": bad}


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_plot(python_exe, script, manifest, pred_dir, sample_id, out_path):
    subprocess.run(
        [
            python_exe,
            script,
            manifest,
            "--pred-dir",
            pred_dir,
            "--id",
            sample_id,
            "--out",
            out_path,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot globally ranked best/middle/worst electronic prediction samples."
    )
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--pred-dir", required=True, help="Prediction .npz directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--sort-metric", choices=METRIC_NAMES, default="band_mae")
    parser.add_argument("--count-per-tier", type=int, default=20)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.count_per_tier < 1:
        raise ValueError("--count-per-tier must be >= 1")

    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "") == args.split]
    if not rows:
        raise ValueError(f"no manifest rows for split={args.split}")

    band_shape, _ = infer_target_layout_from_manifest(args.manifest, rows=rows)
    band_dim = int(np.prod(band_shape)) if band_shape is not None else None

    items = []
    missing = []
    for row in rows:
        item = load_metric_row(row, args.pred_dir, band_dim)
        if item is None:
            missing.append(row.get("id", ""))
        else:
            items.append(item)
    if not items:
        raise ValueError(f"no prediction files found in {args.pred_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    prediction_plot_script = os.path.join(PROJECT_ROOT, "bin", "plot_electronic_prediction.py")
    fitting_plot_script = os.path.join(PROJECT_ROOT, "bin", "plot_electronic_sample_fitting.py")

    selected_by_tier = select_quality_sets(items, args.sort_metric, args.count_per_tier)
    selected_rows = []
    plots_written = 0
    for tier in ("good", "middle", "bad"):
        tier_dir = os.path.join(args.out_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)
        for rank, item in enumerate(selected_by_tier[tier], start=1):
            metric = item.get(args.sort_metric, float("nan"))
            base = f"{rank:02d}_{sanitize_name(item['id'])}_{args.sort_metric}_{metric:.4f}"
            prediction_path = os.path.join(tier_dir, f"{base}_prediction.png")
            fitting_path = os.path.join(tier_dir, f"{base}_fitting.png")
            if not args.skip_existing or not os.path.exists(prediction_path):
                run_plot(args.python_exe, prediction_plot_script, args.manifest, args.pred_dir, item["id"], prediction_path)
                plots_written += 1
            if not args.skip_existing or not os.path.exists(fitting_path):
                run_plot(args.python_exe, fitting_plot_script, args.manifest, args.pred_dir, item["id"], fitting_path)
                plots_written += 1
            selected_rows.append(
                {
                    "tier": tier,
                    "rank_in_tier": rank,
                    "prediction_plot": prediction_path,
                    "fitting_plot": fitting_path,
                    **item,
                }
            )

    fieldnames = [
        "tier",
        "rank_in_tier",
        "id",
        "split",
        "band_path",
        "pred_path",
        "prediction_plot",
        "fitting_plot",
        *METRIC_NAMES,
    ]
    write_csv(os.path.join(args.out_dir, "ranked_selected_samples.csv"), selected_rows, fieldnames)
    write_csv(os.path.join(args.out_dir, "ranked_all_sample_metrics.csv"), items, ["id", "split", "band_path", "pred_path", *METRIC_NAMES])

    print(f"split: {args.split}")
    print(f"manifest rows: {len(rows)}")
    print(f"predictions used: {len(items)}")
    print(f"missing predictions: {len(missing)}")
    print(f"sort metric: {args.sort_metric}")
    print(f"count per tier: {args.count_per_tier}")
    print(f"selected samples: {len(selected_rows)}")
    print(f"plots written: {plots_written}")
    print(f"wrote folder: {args.out_dir}")
    print(f"selected CSV: {os.path.join(args.out_dir, 'ranked_selected_samples.csv')}")


if __name__ == "__main__":
    main()
