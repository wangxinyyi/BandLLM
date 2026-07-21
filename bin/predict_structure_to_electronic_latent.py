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
    GPTConfig,
    StructureToElectronicLatentModel,
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


DECODER_MODULE_NAMES = ("band_head", "band_refine")


def load_decoder_state_dict(autoencoder, state):
    for name in DECODER_MODULE_NAMES:
        module = getattr(autoencoder, name, None)
        if module is None:
            continue
        prefix = f"{name}."
        module_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if module_state:
            module.load_state_dict(module_state)


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


def load_autoencoder_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    ae_config = ElectronicAutoencoderConfig(**checkpoint["config"])
    model = ElectronicAutoencoder(ae_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Predict 16-band lines from CIF-to-latent checkpoints.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--ckpt", required=True, help="structure_to_electronic_latent.pt checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory for prediction .npz files.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "") == args.split]
    dataset = ElectronicStructureDataset(args.manifest, rows=rows)
    if len(dataset) == 0:
        raise ValueError(f"dataset is empty for split={args.split}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    gpt_config = GPTConfig(**checkpoint["gpt_config"])
    latent_dim = int(checkpoint["latent_dim"])
    band_path_to_idx = checkpoint.get("band_path_to_idx", {"unknown": 0})
    model = StructureToElectronicLatentModel(
        gpt_config=gpt_config,
        latent_dim=latent_dim,
        band_path_count=len(band_path_to_idx),
        band_path_emb_dim=int(checkpoint.get("band_path_emb_dim", 64)),
        hidden_dim=int(checkpoint.get("head_hidden_dim", 1024)),
        head_layers=int(checkpoint.get("head_layers", 1)),
        dropout=float(checkpoint.get("head_dropout", 0.1)),
        pooling=checkpoint.get("structure_pooling", "mean"),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    autoencoder_ckpt = resolve_checkpoint_path(
        checkpoint.get("autoencoder_ckpt_abs", "") or checkpoint.get("autoencoder_ckpt", ""),
        checkpoint_path=args.ckpt,
        manifest_path=args.manifest,
    )
    autoencoder, ae_checkpoint = load_autoencoder_checkpoint(autoencoder_ckpt, args.device)
    trained_decoder_path = resolve_checkpoint_path(
        checkpoint.get("trained_decoder_path", ""),
        checkpoint_path=args.ckpt,
        manifest_path=args.manifest,
    )
    if trained_decoder_path and os.path.exists(trained_decoder_path):
        load_decoder_state_dict(autoencoder, torch.load(trained_decoder_path, map_location=args.device))
        autoencoder.eval()

    stats = None
    stats_path = resolve_checkpoint_path(
        checkpoint.get("target_stats_abs", "")
        or checkpoint.get("target_stats", "")
        or ae_checkpoint.get("target_stats_abs", "")
        or ae_checkpoint.get("target_stats", ""),
        checkpoint_path=args.ckpt,
        manifest_path=args.manifest,
    )
    if stats_path:
        loaded = np.load(stats_path)
        stats = {
            "mean": loaded["mean"].astype(np.float32),
            "std": loaded["std"].astype(np.float32),
        }
        if int(stats["mean"].size) > int(autoencoder.config.target_dim):
            print(
                f"target stats dim {stats['mean'].size} exceeds AE target dim {autoencoder.config.target_dim}; "
                "using the leading band dimensions."
            )
            stats["mean"] = stats["mean"][: autoencoder.config.target_dim]
            stats["std"] = stats["std"][: autoencoder.config.target_dim]
        if int(stats["mean"].size) != int(autoencoder.config.target_dim):
            raise ValueError(
                f"target stats dim {stats['mean'].size} does not match AE target dim {autoencoder.config.target_dim}"
            )

    band_shape = tuple(checkpoint.get("band_shape") or ae_checkpoint.get("band_shape") or ()) or None
    dos_dim = checkpoint.get("dos_dim", ae_checkpoint.get("dos_dim"))
    if band_shape is None or dos_dim is None:
        raise ValueError("checkpoint is missing band_shape/dos_dim")

    latent_mean = checkpoint.get("latent_mean")
    latent_std = checkpoint.get("latent_std")
    if latent_mean is not None and latent_std is not None:
        latent_mean = latent_mean.to(device=args.device, dtype=torch.float32)
        latent_std = latent_std.to(device=args.device, dtype=torch.float32)
    else:
        latent_mean = None
        latent_std = None

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
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            band_path_idx = torch.tensor(
                [band_path_to_idx.get(path, 0) for path in batch["band_path"]],
                dtype=torch.long,
                device=args.device,
            )
            target = batch["target"].to(args.device)
            target_mask = batch["target_mask"].to(args.device)

            latent_pred = model(input_ids, attention_mask, band_path_idx=band_path_idx)
            latent_decode = latent_pred if latent_mean is None else latent_pred * latent_std + latent_mean
            pred = autoencoder.decode(latent_decode)

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = target_mask.cpu().numpy()
            latent_np = latent_decode.cpu().numpy()

            if stats is not None:
                pred_np = pred_np * stats["std"] + stats["mean"]
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

                pred_band = pred_np[i][:band_dim]
                target_band = target_np[i][:band_dim]
                mask_band = mask_np[i][:band_dim]
                band_mse = masked_mse(pred_band, target_band, mask_band)
                band_rmse = masked_rmse(pred_band, target_band, mask_band)
                band_mae = masked_mae(pred_band, target_band, mask_band)
                all_band_mse.append(band_mse)
                all_band_rmse.append(band_rmse)
                all_band_mae.append(band_mae)

                np.savez_compressed(
                    os.path.join(args.out_dir, f"{sample_id}.npz"),
                    prediction=pred_np[i].astype(np.float32),
                    latent_prediction=latent_np[i].astype(np.float32),
                    target=target_np[i].astype(np.float32),
                    target_mask=mask_np[i].astype(np.float32),
                    mse=np.asarray(mse, dtype=np.float32),
                    rmse=np.asarray(rmse, dtype=np.float32),
                    mae=np.asarray(mae, dtype=np.float32),
                    band_mse=np.asarray(band_mse, dtype=np.float32),
                    band_rmse=np.asarray(band_rmse, dtype=np.float32),
                    band_mae=np.asarray(band_mae, dtype=np.float32),
                )
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
