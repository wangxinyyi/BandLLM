import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crystallm import read_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description="Assign fixed train/val/test splits to an electronic manifest.")
    parser.add_argument("manifest", help="Input manifest.csv")
    parser.add_argument("--out", required=True, help="Output manifest.csv with split column")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--group-by", choices=["id", "formula", "band_path"], default="id")
    args = parser.parse_args()

    total = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")

    grouped = {}
    for row in rows:
        key = row.get(args.group_by, "") or row.get("id", "")
        grouped.setdefault(key, []).append(row)

    keys = list(grouped.keys())
    random.Random(args.seed).shuffle(keys)

    n = len(keys)
    n_train = max(1, int(round(n * args.train_fraction)))
    n_val = max(1, int(round(n * args.val_fraction))) if n >= 3 else max(0, n - n_train)
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    split_keys = {
        "train": set(keys[:n_train]),
        "val": set(keys[n_train:n_train + n_val]),
        "test": set(keys[n_train + n_val:]),
    }

    out_rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    for key in keys:
        if key in split_keys["train"]:
            split = "train"
        elif key in split_keys["val"]:
            split = "val"
        else:
            split = "test"
        for row in grouped[key]:
            new_row = dict(row)
            new_row["split"] = split
            out_rows.append(new_row)
            counts[split] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_manifest(out_rows, args.out)
    print(f"wrote {args.out}")
    print(f"group_by: {args.group_by}")
    print(f"groups: {n}")
    print(f"samples train/val/test: {counts['train']}/{counts['val']}/{counts['test']}")


if __name__ == "__main__":
    main()
