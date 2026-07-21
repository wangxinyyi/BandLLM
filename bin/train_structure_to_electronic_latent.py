import argparse
import os
import random
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crystallm import (
    ElectronicAutoencoder,
    ElectronicAutoencoderConfig,
    ElectronicStructureDataset,
    GPT,
    GPTConfig,
    StructureToElectronicLatentModel,
    masked_mse,
    masked_reconstruction_loss,
    read_manifest,
    resolve_manifest_path,
)


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


def make_checkpoint_path(path: str, out_dir: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return os.path.abspath(path)


def split_rows(rows, val_fraction: float):
    if any(row.get("split", "") in {"train", "val", "test"} for row in rows):
        train_rows = [row for row in rows if row.get("split", "") == "train"]
        val_rows = [row for row in rows if row.get("split", "") == "val"]
        test_rows = [row for row in rows if row.get("split", "") == "test"]
        if not train_rows:
            raise ValueError("manifest has split labels but no train rows")
        return train_rows, val_rows, test_rows

    shuffled = list(rows)
    random.Random(1337).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[:-val_size] if val_size else shuffled, shuffled[-val_size:] if val_size else [], []


def build_band_path_vocab(rows):
    counts = Counter((row.get("band_path", "") or "unknown") for row in rows)
    vocab = ["unknown"] + sorted(path for path in counts if path and path != "unknown")
    return {path: idx for idx, path in enumerate(vocab)}


def label_type_counts(rows):
    return dict(Counter((row.get("label_type", "") or "full") for row in rows))


DECODER_MODULE_NAMES = ("band_head", "band_refine")


def decoder_parameters(autoencoder):
    for name in DECODER_MODULE_NAMES:
        module = getattr(autoencoder, name, None)
        if module is not None:
            yield from module.parameters()


def set_decoder_train_mode(autoencoder, training: bool):
    for name in DECODER_MODULE_NAMES:
        module = getattr(autoencoder, name, None)
        if module is not None:
            module.train(training)


def decoder_state_dict(autoencoder):
    state = {}
    for name in DECODER_MODULE_NAMES:
        module = getattr(autoencoder, name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value
    return state


def infer_target_layout(sample):
    if "bands" not in sample:
        return None, None
    return tuple(sample["bands"].shape), 0


def transform_target_for_training_torch(target, band_shape, dos_dim):
    if band_shape is None or dos_dim is None or int(dos_dim) <= 0:
        return target
    band_dim = int(band_shape[0] * band_shape[1])
    target = target.clone()
    target[..., band_dim:band_dim + dos_dim] = torch.log1p(
        torch.clamp(target[..., band_dim:band_dim + dos_dim], min=0.0)
    )
    return target


def full_label_mask_from_batch(batch, device, batch_size: int):
    has_band = batch.get("has_band")
    if has_band is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return has_band.to(device=device, dtype=torch.bool)


def masked_latent_loss(latent_pred, latent_target, full_label_mask, loss_type: str = "smooth_l1", beta: float = 1.0):
    if torch.any(full_label_mask):
        pred = latent_pred[full_label_mask]
        target = latent_target[full_label_mask]
        if loss_type in {"smooth_l1", "huber"}:
            return torch.nn.functional.smooth_l1_loss(
                pred,
                target,
                reduction="mean",
                beta=float(beta),
            )
        if loss_type == "mse":
            return masked_mse(pred, target)
        raise ValueError(f"unknown latent loss type: {loss_type}")
    return latent_pred.new_tensor(0.0)


def expand_pretrained_block_size(model: GPT, config: GPTConfig, block_size: int) -> tuple[GPT, GPTConfig]:
    if block_size < config.block_size:
        model.crop_block_size(block_size)
        return model, model.config
    if block_size <= config.block_size:
        return model, config

    expanded_config = GPTConfig(
        block_size=block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias,
    )
    expanded_model = GPT(expanded_config)
    expanded_state = expanded_model.state_dict()
    state = model.state_dict()
    for key, value in state.items():
        if key == "transformer.wpe.weight":
            expanded_state[key][: value.shape[0]] = value
            if value.shape[0] < expanded_state[key].shape[0]:
                tail = expanded_state[key].shape[0] - value.shape[0]
                expanded_state[key][value.shape[0] :] = value[-1:].expand(tail, -1)
        elif key in expanded_state and expanded_state[key].shape == value.shape:
            expanded_state[key] = value
    expanded_model.load_state_dict(expanded_state)
    return expanded_model, expanded_config


def load_pretrained_gpt(path: str, device: str, block_size: int):
    if not path:
        return None, None
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    state = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state.keys()):
        if key.startswith(unwanted_prefix):
            state[key[len(unwanted_prefix):]] = state.pop(key)
    model.load_state_dict(state)
    return expand_pretrained_block_size(model, config, block_size)


def load_autoencoder(path: str, device: str, manifest_path: str):
    checkpoint = torch.load(path, map_location=device)
    config = ElectronicAutoencoderConfig(**checkpoint["config"])
    model = ElectronicAutoencoder(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stats = None
    stats_path = resolve_checkpoint_path(
        checkpoint.get("target_stats_abs", "") or checkpoint.get("target_stats", ""),
        checkpoint_path=path,
        manifest_path=manifest_path,
    )
    if stats_path:
        loaded = np.load(stats_path)
        stats = {
            "mean": torch.from_numpy(loaded["mean"].astype("float32")).to(device),
            "std": torch.from_numpy(loaded["std"].astype("float32")).to(device),
        }
        if int(stats["mean"].numel()) != int(model.config.target_dim):
            if int(stats["mean"].numel()) > int(model.config.target_dim):
                print(
                    f"target stats dim {stats['mean'].numel()} exceeds AE target dim {model.config.target_dim}; "
                    "using the leading band dimensions."
                )
                stats["mean"] = stats["mean"][: model.config.target_dim]
                stats["std"] = stats["std"][: model.config.target_dim]
        if int(stats["mean"].numel()) != int(model.config.target_dim):
            raise ValueError(
                f"target stats dim {stats['mean'].numel()} does not match AE target dim {model.config.target_dim}"
            )
    return model, checkpoint, stats, stats_path


def main():
    parser = argparse.ArgumentParser(description="Train CIF-to-electronic-latent model.")
    parser.add_argument("manifest", help="Prepared electronic dataset manifest.csv.")
    parser.add_argument("--autoencoder-ckpt", required=True, help="electronic_autoencoder.pt checkpoint.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pretrained-ckpt", default="", help="Optional CrystaLLM ckpt.pt.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=0.0, help="Optional decoder LR when decoder is not frozen. 0 uses lr * decoder-lr-mult.")
    parser.add_argument("--decoder-lr-mult", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--freeze-gpt", action="store_true")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--structure-pooling", choices=["mean", "last", "attn"], default="attn")
    parser.add_argument("--band-path-emb-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=2048)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--latent-loss-type", choices=["mse", "smooth_l1", "huber"], default="smooth_l1")
    parser.add_argument("--latent-huber-beta", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--recon-loss-weight", type=float, default=1.0)
    parser.add_argument("--recon-start-epoch", type=int, default=5)
    parser.add_argument("--recon-warmup-epochs", type=int, default=10)
    parser.add_argument("--band-weight", type=float, default=1.0)
    parser.add_argument("--dos-weight", type=float, default=0.0, help="Ignored for band-only targets.")
    parser.add_argument("--band-slope-weight", type=float, default=0.05)
    parser.add_argument("--dos-slope-weight", type=float, default=0.0, help="Ignored for band-only targets.")
    parser.add_argument("--band-curvature-weight", type=float, default=0.0)
    parser.add_argument("--band-fermi-window", type=float, default=0.0, help="Fermi-window half width in eV. 0 disables it.")
    parser.add_argument("--band-fermi-weight", type=float, default=0.0)
    parser.add_argument("--band-gap-weight", type=float, default=0.0)
    parser.add_argument("--band-gap-tol", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--latent-stats-batch-size", type=int, default=64)
    parser.add_argument("--latent-std-floor", type=float, default=1e-4)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=371)
    parser.add_argument("--strict-block-size", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")
    train_rows, val_rows, _ = split_rows(rows, args.val_fraction)
    train_set = ElectronicStructureDataset(args.manifest, rows=train_rows)
    val_set = ElectronicStructureDataset(args.manifest, rows=val_rows) if val_rows else None
    sample = train_set[0]
    band_shape, dos_dim = infer_target_layout(sample)
    target_dim = int(sample["target"].numel())
    print(f"label_type counts train={label_type_counts(train_rows)} val={label_type_counts(val_rows)}")
    print(f"target representation: band16_only_v1 shape={band_shape}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers) if val_set is not None else None

    autoencoder_ckpt_path = os.path.abspath(args.autoencoder_ckpt)
    autoencoder, ae_checkpoint, stats, stats_path = load_autoencoder(autoencoder_ckpt_path, args.device, args.manifest)
    if int(autoencoder.config.target_dim) != target_dim:
        raise ValueError(f"AE target dim {autoencoder.config.target_dim} does not match dataset target dim {target_dim}")
    latent_dim = int(autoencoder.config.latent_dim)

    if args.freeze_decoder:
        for param in decoder_parameters(autoencoder):
            param.requires_grad = False

    pretrained_ckpt_path = os.path.abspath(args.pretrained_ckpt) if args.pretrained_ckpt else ""
    if pretrained_ckpt_path:
        pre_ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        pretrained_block_size = int(pre_ckpt["model_args"]["block_size"])
        if args.strict_block_size and args.block_size > pretrained_block_size:
            raise ValueError(
                f"--block-size={args.block_size} exceeds pretrained block_size={pretrained_block_size}"
            )
    pretrained, gpt_config = load_pretrained_gpt(pretrained_ckpt_path, args.device, args.block_size)
    if args.freeze_gpt and pretrained is None:
        raise ValueError("--freeze-gpt requires --pretrained-ckpt")
    if gpt_config is None:
        gpt_config = GPTConfig(
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            bias=False,
        )

    band_path_to_idx = build_band_path_vocab(train_rows)
    model = StructureToElectronicLatentModel(
        gpt_config=gpt_config,
        latent_dim=latent_dim,
        band_path_count=len(band_path_to_idx),
        band_path_emb_dim=args.band_path_emb_dim,
        hidden_dim=args.head_hidden_dim,
        head_layers=args.head_layers,
        dropout=args.head_dropout,
        pretrained_gpt=pretrained,
        pooling=args.structure_pooling,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        "structure model: "
        f"params={n_params / 1e6:.2f}M "
        f"head_hidden={args.head_hidden_dim} head_layers={args.head_layers} "
        f"head_dropout={args.head_dropout} band_path_emb={args.band_path_emb_dim} "
        f"latent_loss={args.latent_loss_type} beta={args.latent_huber_beta}"
    )

    if args.freeze_gpt:
        for param in model.gpt.parameters():
            param.requires_grad = False

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    decoder_trainable_parameters = [p for p in decoder_parameters(autoencoder) if p.requires_grad]
    parameters = model_parameters + decoder_trainable_parameters
    optimizer_groups = [{"params": model_parameters, "lr": args.lr}]
    if decoder_trainable_parameters:
        decoder_lr = args.decoder_lr if args.decoder_lr > 0 else args.lr * args.decoder_lr_mult
        optimizer_groups.append({"params": decoder_trainable_parameters, "lr": decoder_lr})
    else:
        decoder_lr = 0.0
    optimizer = torch.optim.AdamW(optimizer_groups)
    print(f"optimizer: model_lr={args.lr} decoder_lr={decoder_lr} trainable_decoder={len(decoder_trainable_parameters) > 0}")
    print(
        "recon loss weights: "
        f"band={args.band_weight} slope={args.band_slope_weight} curvature={args.band_curvature_weight} "
        f"fermi_window={args.band_fermi_window} fermi={args.band_fermi_weight} gap={args.band_gap_weight}"
    )
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    def normalize_target(target):
        target = transform_target_for_training_torch(target, band_shape, dos_dim)
        if stats is None:
            return target
        return (target - stats["mean"]) / stats["std"]

    def compute_latent_stats():
        stats_loader = DataLoader(
            train_set,
            batch_size=args.latent_stats_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        sum_x = torch.zeros(latent_dim, dtype=torch.float64, device=args.device)
        sum_x2 = torch.zeros(latent_dim, dtype=torch.float64, device=args.device)
        count = 0
        autoencoder.eval()
        with torch.no_grad():
            for batch in tqdm(stats_loader, desc="latent stats"):
                target = batch["target"].to(args.device)
                full_label_mask = full_label_mask_from_batch(batch, args.device, target.shape[0])
                if not torch.any(full_label_mask):
                    continue
                latent = autoencoder.encode(normalize_target(target))[full_label_mask].to(dtype=torch.float64)
                sum_x += latent.sum(dim=0)
                sum_x2 += (latent * latent).sum(dim=0)
                count += int(latent.shape[0])
        if count <= 0:
            raise ValueError("cannot compute latent stats because no full-label training samples were found")
        mean = sum_x / float(count)
        var = torch.clamp(sum_x2 / float(count) - mean * mean, min=0.0)
        std = torch.sqrt(var).clamp_min(float(args.latent_std_floor))
        mean = mean.to(dtype=torch.float32)
        std = std.to(dtype=torch.float32)
        print(
            "latent stats: "
            f"count={count} "
            f"mean_abs={float(mean.abs().mean().item()):.6f} "
            f"std_mean={float(std.mean().item()):.6f} "
            f"std_min={float(std.min().item()):.6f} "
            f"std_max={float(std.max().item()):.6f}"
        )
        return mean, std

    latent_mean, latent_std = compute_latent_stats()

    def normalize_latent(latent):
        return (latent - latent_mean) / latent_std

    def denormalize_latent(latent):
        return latent * latent_std + latent_mean

    best_val = float("inf")
    for epoch in range(args.epochs):
        if epoch < args.recon_start_epoch:
            effective_recon_weight = 0.0
        elif args.recon_warmup_epochs > 0:
            warmup_step = epoch - args.recon_start_epoch + 1
            effective_recon_weight = args.recon_loss_weight * min(1.0, warmup_step / args.recon_warmup_epochs)
        else:
            effective_recon_weight = args.recon_loss_weight
        model.train()
        set_decoder_train_mode(autoencoder, not args.freeze_decoder)
        train_loss = 0.0
        train_latent_loss = 0.0
        train_recon_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch} train"), start=1):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            band_path_idx = torch.tensor(
                [band_path_to_idx.get(path, 0) for path in batch["band_path"]],
                dtype=torch.long,
                device=args.device,
            )
            target = batch["target"].to(args.device)
            target_mask = batch["target_mask"].to(args.device)
            full_label_mask = full_label_mask_from_batch(batch, args.device, target.shape[0])
            norm_target = normalize_target(target)

            with torch.no_grad():
                latent_target = normalize_latent(autoencoder.encode(norm_target))

            latent_pred = model(input_ids, attention_mask, band_path_idx=band_path_idx)
            recon_pred = autoencoder.decode(denormalize_latent(latent_pred))
            latent_loss = masked_latent_loss(
                latent_pred,
                latent_target,
                full_label_mask,
                loss_type=args.latent_loss_type,
                beta=args.latent_huber_beta,
            )
            recon_loss = masked_reconstruction_loss(
                recon_pred,
                norm_target,
                mask=target_mask,
                band_shape=band_shape,
                dos_dim=dos_dim,
                band_weight=args.band_weight,
                dos_weight=args.dos_weight,
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
            loss = args.latent_loss_weight * latent_loss + effective_recon_weight * recon_loss
            (loss / args.grad_accum_steps).backward()
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_((p for p in parameters if p.requires_grad), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item()
            train_latent_loss += latent_loss.item()
            train_recon_loss += recon_loss.item()
        train_loss /= max(1, len(train_loader))
        train_latent_loss /= max(1, len(train_loader))
        train_recon_loss /= max(1, len(train_loader))

        val_loss = train_loss
        val_latent_loss = train_latent_loss
        val_recon_loss = train_recon_loss
        if val_loader is not None:
            model.eval()
            set_decoder_train_mode(autoencoder, False)
            val_loss = 0.0
            val_latent_loss = 0.0
            val_recon_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"epoch {epoch} val"):
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)
                    band_path_idx = torch.tensor(
                        [band_path_to_idx.get(path, 0) for path in batch["band_path"]],
                        dtype=torch.long,
                        device=args.device,
                    )
                    target = batch["target"].to(args.device)
                    target_mask = batch["target_mask"].to(args.device)
                    full_label_mask = full_label_mask_from_batch(batch, args.device, target.shape[0])
                    norm_target = normalize_target(target)
                    latent_target = normalize_latent(autoencoder.encode(norm_target))
                    latent_pred = model(input_ids, attention_mask, band_path_idx=band_path_idx)
                    latent_loss = masked_latent_loss(
                        latent_pred,
                        latent_target,
                        full_label_mask,
                        loss_type=args.latent_loss_type,
                        beta=args.latent_huber_beta,
                    )
                    recon_pred = autoencoder.decode(denormalize_latent(latent_pred))
                    recon_loss = masked_reconstruction_loss(
                        recon_pred,
                        norm_target,
                        mask=target_mask,
                        band_shape=band_shape,
                        dos_dim=dos_dim,
                        band_weight=args.band_weight,
                        dos_weight=args.dos_weight,
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
                    val_loss += (args.latent_loss_weight * latent_loss + effective_recon_weight * recon_loss).item()
                    val_latent_loss += latent_loss.item()
                    val_recon_loss += recon_loss.item()
            val_loss /= max(1, len(val_loader))
            val_latent_loss /= max(1, len(val_loader))
            val_recon_loss /= max(1, len(val_loader))

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_latent={train_latent_loss:.6f} val_latent={val_latent_loss:.6f} "
            f"train_recon={train_recon_loss:.6f} val_recon={val_recon_loss:.6f} "
            f"recon_weight={effective_recon_weight:.6f}"
        )
        checkpoint_score = float(args.latent_loss_weight) * val_latent_loss + float(args.recon_loss_weight) * val_recon_loss
        checkpoint = {
            "model": model.state_dict(),
            "gpt_config": gpt_config.__dict__,
            "autoencoder_ckpt": make_checkpoint_path(autoencoder_ckpt_path, args.out_dir),
            "autoencoder_ckpt_abs": autoencoder_ckpt_path,
            "target_stats": make_checkpoint_path(stats_path, args.out_dir) if stats_path else "",
            "target_stats_abs": os.path.abspath(stats_path) if stats_path else "",
            "latent_dim": latent_dim,
            "latent_mean": latent_mean.detach().cpu(),
            "latent_std": latent_std.detach().cpu(),
            "latent_standardized": True,
            "structure_pooling": args.structure_pooling,
            "band_path_to_idx": band_path_to_idx,
            "band_path_emb_dim": args.band_path_emb_dim,
            "head_hidden_dim": args.head_hidden_dim,
            "head_layers": args.head_layers,
            "head_dropout": args.head_dropout,
            "latent_loss_type": args.latent_loss_type,
            "latent_huber_beta": args.latent_huber_beta,
            "target_representation": ae_checkpoint.get("target_representation", "band16_only_v1"),
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
            "latent_loss_weight": args.latent_loss_weight,
            "recon_loss_weight": args.recon_loss_weight,
            "recon_start_epoch": args.recon_start_epoch,
            "recon_warmup_epochs": args.recon_warmup_epochs,
            "decoder_lr": decoder_lr,
            "decoder_lr_mult": args.decoder_lr_mult,
            "max_grad_norm": args.max_grad_norm,
            "latent_stats_batch_size": args.latent_stats_batch_size,
            "latent_std_floor": args.latent_std_floor,
            "freeze_decoder": args.freeze_decoder,
            "trained_decoder_path": "" if args.freeze_decoder else "structure_to_electronic_latent_decoder.pt",
            "partial_label_training": True,
            "dos_transform": "none",
            "epoch": epoch,
            "val_loss": val_loss,
            "val_latent_loss": val_latent_loss,
            "val_recon_loss": val_recon_loss,
            "checkpoint_score": checkpoint_score,
        }
        torch.save(checkpoint, os.path.join(args.out_dir, "structure_to_electronic_latent_latest.pt"))
        if checkpoint_score <= best_val:
            best_val = checkpoint_score
            torch.save(checkpoint, os.path.join(args.out_dir, "structure_to_electronic_latent.pt"))
            if not args.freeze_decoder:
                torch.save(
                    decoder_state_dict(autoencoder),
                    os.path.join(args.out_dir, "structure_to_electronic_latent_decoder.pt"),
                )


if __name__ == "__main__":
    main()
