import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from crystallm import (
    ElectronicAutoencoder,
    ElectronicAutoencoderConfig,
    ElectronicStructureDataset,
    inverse_target_transform,
    read_manifest,
    resolve_manifest_path,
)


def masked_mae(pred, target, mask):
    diff = np.abs(pred - target) * mask
    return float(diff.sum() / max(mask.sum(), 1.0))


def masked_mse(pred, target, mask):
    diff = ((pred - target) ** 2) * mask
    return float(diff.sum() / max(mask.sum(), 1.0))


def masked_rmse(pred, target, mask):
    return float(np.sqrt(max(masked_mse(pred, target, mask), 0.0)))


def resolve_checkpoint_path(path: str, checkpoint_path: str, manifest_path: str = "") -> str:
    if not path:
        return ""
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        if checkpoint_path:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), path))
        if manifest_path:
            candidates.append(resolve_manifest_path(manifest_path, path))
        candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def transform_target_for_training_torch(target, band_shape, dos_dim):
    if band_shape is None or dos_dim is None or int(dos_dim) <= 0:
        return target
    band_dim = int(band_shape[0] * band_shape[1])
    target = target.clone()
    target[..., band_dim:band_dim + dos_dim] = torch.log1p(
        torch.clamp(target[..., band_dim:band_dim + dos_dim], min=0.0)
    )
    return target


def load_autoencoder_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    config = ElectronicAutoencoderConfig(**checkpoint["config"])
    model = ElectronicAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct electronic targets with the Stage-I autoencoder and report upper-bound metrics."
    )
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--ckpt", required=True, help="electronic_autoencoder.pt checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory for reconstruction .npz files.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Evaluate only one fixed split when manifest has a split column.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "") == args.split]
    dataset = ElectronicStructureDataset(args.manifest, rows=rows)
    if len(dataset) == 0:
        raise ValueError(f"dataset is empty for split={args.split}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model, checkpoint = load_autoencoder_checkpoint(args.ckpt, args.device)

    stats = None
    stats_path = resolve_checkpoint_path(
        checkpoint.get("target_stats_abs", "") or checkpoint.get("target_stats", ""),
        checkpoint_path=args.ckpt,
        manifest_path=args.manifest,
    )
    band_shape = tuple(checkpoint.get("band_shape") or ()) or None
    dos_dim = checkpoint.get("dos_dim")
    if stats_path:
        loaded = np.load(stats_path)
        stats = {
            "mean": torch.from_numpy(loaded["mean"].astype("float32")).to(args.device),
            "std": torch.from_numpy(loaded["std"].astype("float32")).to(args.device),
        }
        if int(stats["mean"].numel()) != int(model.config.target_dim):
            if int(stats["mean"].numel()) > int(model.config.target_dim):
                print(
                    f"target stats dim {stats['mean'].numel()} exceeds checkpoint target dim "
                    f"{model.config.target_dim}; using the leading band dimensions."
                )
                stats["mean"] = stats["mean"][: model.config.target_dim]
                stats["std"] = stats["std"][: model.config.target_dim]
        if int(stats["mean"].numel()) != int(model.config.target_dim):
            raise ValueError(
                f"target stats dim {stats['mean'].numel()} does not match checkpoint target dim {model.config.target_dim}"
            )

    all_mse = []
    all_rmse = []
    all_mae = []
    all_band_mse = []
    all_band_rmse = []
    all_band_mae = []
    offset = 0
    band_dim = int(np.prod(band_shape))

    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(args.device)
            target_mask = batch["target_mask"].to(args.device)

            norm_target = transform_target_for_training_torch(target, band_shape, dos_dim)
            if stats is not None:
                norm_target = (norm_target - stats["mean"]) / stats["std"]

            recon, latent, _ = model(
                norm_target,
                target_mask,
                band_shape=band_shape,
                dos_dim=dos_dim,
            )

            pred_np = recon.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = target_mask.cpu().numpy()
            latent_np = latent.cpu().numpy()

            if stats is not None:
                pred_np = pred_np * stats["std"].cpu().numpy() + stats["mean"].cpu().numpy()
            pred_np = inverse_target_transform(pred_np, band_shape, dos_dim, clamp_dos=True)

            for i in range(pred_np.shape[0]):
                row = dataset.rows[offset + i]
                sample_id = row.get("id", str(offset + i))
                mse = masked_mse(pred_np[i], target_np[i], mask_np[i])
                rmse = masked_rmse(pred_np[i], target_np[i], mask_np[i])
                mae = masked_mae(pred_np[i], target_np[i], mask_np[i])
                all_mse.append(mse)
                all_rmse.append(rmse)
                all_mae.append(mae)

                band_dim = int(np.prod(band_shape))
                pred_band = pred_np[i][:band_dim]
                target_band = target_np[i][:band_dim]
                mask_band = mask_np[i][:band_dim]

                band_mse = masked_mse(pred_band, target_band, mask_band)
                band_rmse = masked_rmse(pred_band, target_band, mask_band)
                band_mae = masked_mae(pred_band, target_band, mask_band)

                all_band_mse.append(band_mse)
                all_band_rmse.append(band_rmse)
                all_band_mae.append(band_mae)

                sample_npz = {
                    "prediction": pred_np[i].astype(np.float32),
                    "latent_prediction": latent_np[i].astype(np.float32),
                    "target": target_np[i].astype(np.float32),
                    "target_mask": mask_np[i].astype(np.float32),
                    "mse": np.asarray(mse, dtype=np.float32),
                    "rmse": np.asarray(rmse, dtype=np.float32),
                    "mae": np.asarray(mae, dtype=np.float32),
                    "band_mse": np.asarray(band_mse, dtype=np.float32),
                    "band_rmse": np.asarray(band_rmse, dtype=np.float32),
                    "band_mae": np.asarray(band_mae, dtype=np.float32),
                }
                np.savez_compressed(os.path.join(args.out_dir, f"{sample_id}.npz"), **sample_npz)
            offset += pred_np.shape[0]

    print(f"split: {args.split}")
    print(f"samples: {len(dataset)}")
    print("band_mse(eV^2)\tband_rmse(eV)\tband_mae(eV)")
    print(
        f"{float(np.mean(all_band_mse)):.6f}\t"
        f"{float(np.mean(all_band_rmse)):.6f}\t"
        f"{float(np.mean(all_band_mae)):.6f}"
    )
    print(f"wrote predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
