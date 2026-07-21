import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from crystallm import (
    ElectronicAutoencoder,
    ElectronicAutoencoderConfig,
    ElectronicStructureDataset,
    read_manifest,
    resolve_manifest_path,
)


def infer_target_layout(manifest_path: str, sample) -> tuple[tuple[int, int] | None, int | None]:
    if "bands" in sample:
        return tuple(sample["bands"].shape), 0

    rows = read_manifest(manifest_path)
    if not rows:
        return None, None
    root = os.path.dirname(os.path.abspath(manifest_path))
    sample_path = rows[0]["sample_path"]
    path = sample_path if os.path.isabs(sample_path) else os.path.join(root, sample_path)
    with np.load(path) as data:
        if "band16" in data.files:
            return tuple(data["band16"].shape), 0
        if "bands" in data.files:
            return tuple(sample["bands"].shape), 0
    return None, None


def transform_target_for_training_torch(target, band_shape, dos_dim):
    if band_shape is None or dos_dim is None or int(dos_dim) <= 0:
        return target
    band_dim = int(band_shape[0] * band_shape[1])
    target = target.clone()
    target[..., band_dim:band_dim + dos_dim] = torch.log1p(torch.clamp(target[..., band_dim:band_dim + dos_dim], min=0.0))
    return target


def resolve_optional_path(path: str, manifest_path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        resolve_manifest_path(manifest_path, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def make_checkpoint_path(path: str, out_dir: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return os.path.abspath(path)


def split_rows(rows, val_fraction: float):
    tagged = [row for row in rows if row.get("split", "") in {"train", "val", "test"}]
    if tagged:
        train_rows = [row for row in rows if row.get("split", "") == "train"]
        val_rows = [row for row in rows if row.get("split", "") == "val"]
        if not train_rows:
            raise ValueError("manifest has split labels but no train rows")
        return train_rows, val_rows
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Train a band-only autoencoder for 16-band tensors.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--target-stats", default="", help="Optional .npz with target mean/std.")
    parser.add_argument("--band-weight", type=float, default=1.0)
    parser.add_argument("--dos-weight", type=float, default=0.0, help="Ignored for band-only targets.")
    parser.add_argument("--band-slope-weight", type=float, default=0.10)
    parser.add_argument("--dos-slope-weight", type=float, default=0.0, help="Ignored for band-only targets.")
    parser.add_argument("--band-curvature-weight", type=float, default=0.0)
    parser.add_argument("--band-fermi-window", type=float, default=0.0, help="Fermi-window half width in eV. 0 disables it.")
    parser.add_argument("--band-fermi-weight", type=float, default=0.0)
    parser.add_argument("--band-gap-weight", type=float, default=0.0)
    parser.add_argument("--band-gap-tol", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    train_rows, val_rows = split_rows(rows, args.val_fraction)
    if train_rows is not None:
        train_set = ElectronicStructureDataset(args.manifest, rows=train_rows)
        val_set = ElectronicStructureDataset(args.manifest, rows=val_rows) if val_rows else None
        sample = train_set[0]
        print(f"using fixed manifest split: train={len(train_set)} val={len(val_set) if val_set is not None else 0}")
    else:
        dataset = ElectronicStructureDataset(args.manifest)
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        sample = dataset[0]
        val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size > 0:
            train_set, val_set = random_split(dataset, [train_size, val_size])
        else:
            train_set, val_set = dataset, None
        print(f"using random split: train={len(train_set)} val={len(val_set) if val_set is not None else 0}")
    if len(train_set) == 0:
        raise ValueError("training set is empty")
    target_dim = int(sample["target"].numel())
    band_shape, dos_dim = infer_target_layout(args.manifest, sample)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers) if val_set is not None else None

    config = ElectronicAutoencoderConfig(
        target_dim=target_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        band_kpoints=band_shape[0],
        band_count=band_shape[1],
        dos_dim=dos_dim,
    )
    model = ElectronicAutoencoder(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        "model: "
        f"target_dim={target_dim} band_shape={band_shape} "
        f"latent_dim={args.latent_dim} hidden_dim={args.hidden_dim} dropout={args.dropout} "
        f"params={n_params / 1e6:.2f}M"
    )
    print(
        "loss weights: "
        f"band={args.band_weight} band_slope={args.band_slope_weight} "
        f"band_curvature={args.band_curvature_weight} "
        f"fermi_window={args.band_fermi_window} fermi={args.band_fermi_weight} "
        f"gap={args.band_gap_weight} "
        f"lr={args.lr} weight_decay={args.weight_decay}"
    )

    stats = None
    resolved_target_stats = resolve_optional_path(args.target_stats, args.manifest)
    if resolved_target_stats:
        loaded = np.load(resolved_target_stats)
        stats = {
            "mean": torch.from_numpy(loaded["mean"].astype("float32")).to(args.device),
            "std": torch.from_numpy(loaded["std"].astype("float32")).to(args.device),
        }
        if int(stats["mean"].numel()) > target_dim:
            print(
                f"target stats dim {stats['mean'].numel()} exceeds band-only target dim {target_dim}; "
                "using the leading band dimensions."
            )
            stats["mean"] = stats["mean"][:target_dim]
            stats["std"] = stats["std"][:target_dim]
        if int(stats["mean"].numel()) != target_dim:
            raise ValueError(
                f"target stats dim {stats['mean'].numel()} does not match dataset target dim {target_dim}. "
                "Recompute band-only target stats."
            )

    def normalize_target(target):
        target = transform_target_for_training_torch(target, band_shape, dos_dim)
        if stats is None:
            return target
        return (target - stats["mean"]) / stats["std"]

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch} train"):
            target = batch["target"].to(args.device)
            mask = batch["target_mask"].to(args.device)
            target = normalize_target(target)
            _, _, loss = model(
                target,
                mask,
                band_shape=band_shape,
                dos_dim=dos_dim,
                dos_weight=args.dos_weight,
                band_weight=args.band_weight,
                band_slope_weight=args.band_slope_weight,
                dos_slope_weight=args.dos_slope_weight,
                band_curvature_weight=args.band_curvature_weight,
                band_fermi_window=args.band_fermi_window,
                band_fermi_weight=args.band_fermi_weight,
                band_gap_weight=args.band_gap_weight,
                band_gap_tol=args.band_gap_tol,
                target_mean=stats["mean"] if stats is not None else None,
                target_std=stats["std"] if stats is not None else None,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                    target = batch["target"].to(args.device)
                    mask = batch["target_mask"].to(args.device)
                    target = normalize_target(target)
                    _, _, loss = model(
                        target,
                        mask,
                        band_shape=band_shape,
                        dos_dim=dos_dim,
                        dos_weight=args.dos_weight,
                        band_weight=args.band_weight,
                        band_slope_weight=args.band_slope_weight,
                        dos_slope_weight=args.dos_slope_weight,
                        band_curvature_weight=args.band_curvature_weight,
                        band_fermi_window=args.band_fermi_window,
                        band_fermi_weight=args.band_fermi_weight,
                        band_gap_weight=args.band_gap_weight,
                        band_gap_tol=args.band_gap_tol,
                        target_mean=stats["mean"] if stats is not None else None,
                        target_std=stats["std"] if stats is not None else None,
                    )
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "target_stats": make_checkpoint_path(resolved_target_stats, args.out_dir),
                    "target_stats_abs": os.path.abspath(resolved_target_stats) if resolved_target_stats else "",
                    "manifest_path": os.path.abspath(args.manifest),
                    "target_representation": "band16_only_v1",
                    "band_shape": band_shape,
                    "dos_dim": dos_dim,
                    "band_weight": args.band_weight,
                    "dos_weight": args.dos_weight,
                    "band_slope_weight": args.band_slope_weight,
                    "dos_slope_weight": args.dos_slope_weight,
                    "band_curvature_weight": args.band_curvature_weight,
                    "band_fermi_window": args.band_fermi_window,
                    "band_fermi_weight": args.band_fermi_weight,
                    "band_gap_weight": args.band_gap_weight,
                    "band_gap_tol": args.band_gap_tol,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "dos_transform": "none",
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                os.path.join(args.out_dir, "electronic_autoencoder.pt"),
            )


if __name__ == "__main__":
    main()
