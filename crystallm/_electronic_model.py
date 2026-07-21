from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from crystallm import GPT, GPTConfig


@dataclass
class ElectronicAutoencoderConfig:
    target_dim: int
    latent_dim: int = 1024
    hidden_dim: int = 4096
    dropout: float = 0.05
    band_kpoints: int = 128
    band_count: int = 16
    dos_dim: int = 0


class _DilatedBandRefiner(nn.Module):
    def __init__(self, band_count: int, width: int = 128, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Conv1d(band_count, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation),
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, band_count, kernel_size=1),
        )

    def forward(self, band_channels: Tensor) -> Tensor:
        hidden = self.in_proj(band_channels)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.out(hidden)


class _DilatedDosRefiner(nn.Module):
    def __init__(self, width: int = 128, dropout: float = 0.05):
        super().__init__()
        self.in_proj = nn.Conv1d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(width, width, kernel_size=9, padding=4 * dilation, dilation=dilation),
                )
                for dilation in (1, 2, 4, 8, 16)
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, dos_channels: Tensor) -> Tensor:
        hidden = self.in_proj(dos_channels)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.out(hidden)


class ElectronicAutoencoder(nn.Module):
    def __init__(self, config: ElectronicAutoencoderConfig):
        super().__init__()
        self.config = config
        band_dim = int(config.band_kpoints * config.band_count)
        expected_dim = band_dim + max(0, int(config.dos_dim))
        if int(config.target_dim) != expected_dim:
            raise ValueError(f"target_dim={config.target_dim} does not match configured target dim={expected_dim}")

        self.encoder = nn.Sequential(
            nn.LayerNorm(config.target_dim),
            nn.Linear(config.target_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.band_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, band_dim),
        )
        self.band_refine = _DilatedBandRefiner(config.band_count, width=128, dropout=config.dropout)

        if int(config.dos_dim) > 0:
            self.dos_head = nn.Sequential(
                nn.LayerNorm(config.latent_dim),
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.dos_dim),
            )
            self.dos_refine = _DilatedDosRefiner(width=128, dropout=config.dropout)
        else:
            self.dos_head = None
            self.dos_refine = None

    @property
    def band_shape(self) -> Tuple[int, int]:
        return int(self.config.band_kpoints), int(self.config.band_count)

    def encode(self, target: Tensor) -> Tensor:
        return self.encoder(target)

    def decode(self, latent: Tensor) -> Tensor:
        batch = latent.shape[0]
        band_kpoints, band_count = self.band_shape
        band_seed = self.band_head(latent).reshape(batch, band_kpoints, band_count)
        band_channels = band_seed.permute(0, 2, 1)
        band = (band_channels + self.band_refine(band_channels)).permute(0, 2, 1)
        if int(self.config.dos_dim) <= 0:
            return band.reshape(batch, -1)
        dos_seed = self.dos_head(latent)
        dos = dos_seed + self.dos_refine(dos_seed.unsqueeze(1)).squeeze(1)
        return torch.cat([band.reshape(batch, -1), dos], dim=-1)

    def forward(
        self,
        target: Tensor,
        mask: Optional[Tensor] = None,
        band_shape: Optional[Tuple[int, int]] = None,
        dos_dim: Optional[int] = None,
        band_weight: float = 1.0,
        dos_weight: float = 1.0,
        band_slope_weight: float = 0.05,
        dos_slope_weight: float = 0.05,
        band_curvature_weight: float = 0.0,
        band_fermi_window: float = 0.0,
        band_fermi_weight: float = 0.0,
        band_gap_weight: float = 0.0,
        band_gap_tol: float = 1e-3,
        target_mean: Optional[Tensor] = None,
        target_std: Optional[Tensor] = None,
    ):
        latent = self.encode(target)
        recon = self.decode(latent)
        loss = masked_reconstruction_loss(
            recon,
            target,
            mask=mask,
            band_shape=band_shape or self.band_shape,
            dos_dim=int(self.config.dos_dim) if dos_dim is None else dos_dim,
            band_weight=band_weight,
            dos_weight=dos_weight,
            band_slope_weight=band_slope_weight,
            dos_slope_weight=dos_slope_weight,
            band_curvature_weight=band_curvature_weight,
            band_fermi_window=band_fermi_window,
            band_fermi_weight=band_fermi_weight,
            band_gap_weight=band_gap_weight,
            band_gap_tol=band_gap_tol,
            target_mean=target_mean,
            target_std=target_std,
        )
        return recon, latent, loss


class _PooledStructureEncoder(nn.Module):
    def __init__(
        self,
        gpt_config: GPTConfig,
        pretrained_gpt: Optional[GPT] = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.gpt = pretrained_gpt if pretrained_gpt is not None else GPT(gpt_config)
        if pooling not in {"mean", "last", "attn"}:
            raise ValueError(f"unknown structure pooling: {pooling}")
        self.pooling = pooling
        self.pool_score = nn.Linear(gpt_config.n_embd, 1) if pooling == "attn" else None

    def _pool_tokens(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        if attention_mask is None:
            if self.pooling == "last":
                return x[:, -1, :]
            if self.pooling == "attn" and self.pool_score is not None:
                weights = torch.softmax(self.pool_score(x).squeeze(-1), dim=-1)
                return (x * weights.unsqueeze(-1)).sum(dim=1)
            return x.mean(dim=1)

        mask = attention_mask[:, : x.size(1)].to(dtype=x.dtype)
        if self.pooling == "last":
            lengths = mask.sum(dim=1).to(dtype=torch.long).clamp_min(1)
            batch_idx = torch.arange(x.size(0), device=x.device)
            return x[batch_idx, lengths - 1, :]

        if self.pooling == "attn" and self.pool_score is not None:
            scores = self.pool_score(x).squeeze(-1)
            scores = scores.masked_fill(mask <= 0, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return (x * weights.unsqueeze(-1)).sum(dim=1)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def encode_tokens(self, idx: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        device = idx.device
        _, t = idx.size()
        if t > self.gpt.config.block_size:
            idx = idx[:, : self.gpt.config.block_size]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.gpt.config.block_size]
            t = idx.size(1)

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.gpt.transformer.wte(idx)
        pos_emb = self.gpt.transformer.wpe(pos)
        x = self.gpt.transformer.drop(tok_emb + pos_emb)
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        return self._pool_tokens(x, attention_mask)


class StructureToElectronicModel(_PooledStructureEncoder):
    def __init__(
        self,
        gpt_config: GPTConfig,
        target_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        pretrained_gpt: Optional[GPT] = None,
        pooling: str = "mean",
    ):
        super().__init__(gpt_config=gpt_config, pretrained_gpt=pretrained_gpt, pooling=pooling)
        self.head = nn.Sequential(
            nn.LayerNorm(gpt_config.n_embd),
            nn.Linear(gpt_config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(
        self,
        idx: Tensor,
        attention_mask: Optional[Tensor],
        target: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        pred = self.head(self.encode_tokens(idx, attention_mask))
        loss = masked_mse(pred, target, mask) if target is not None else None
        return pred, loss


class StructureToElectronicLatentModel(_PooledStructureEncoder):
    def __init__(
        self,
        gpt_config: GPTConfig,
        latent_dim: int,
        band_path_count: int = 0,
        band_path_emb_dim: int = 64,
        hidden_dim: int = 1024,
        head_layers: int = 1,
        dropout: float = 0.1,
        pretrained_gpt: Optional[GPT] = None,
        pooling: str = "mean",
    ):
        super().__init__(gpt_config=gpt_config, pretrained_gpt=pretrained_gpt, pooling=pooling)
        self.band_path_emb = None
        fused_dim = gpt_config.n_embd
        if band_path_count > 0:
            self.band_path_emb = nn.Embedding(band_path_count, band_path_emb_dim)
            fused_dim += band_path_emb_dim
        self.fused_dim = fused_dim
        layers = [nn.LayerNorm(fused_dim)]
        in_dim = fused_dim
        for layer_idx in range(max(1, int(head_layers))):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
            if layer_idx + 1 < max(1, int(head_layers)):
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.head = nn.Sequential(*layers)

    def encode_condition(self, idx: Tensor, attention_mask: Optional[Tensor], band_path_idx: Optional[Tensor] = None) -> Tensor:
        pooled = self.encode_tokens(idx, attention_mask)
        if self.band_path_emb is not None:
            if band_path_idx is None:
                raise ValueError("band_path_idx is required when band_path_count > 0")
            pooled = torch.cat([pooled, self.band_path_emb(band_path_idx)], dim=-1)
        return pooled

    def forward(self, idx: Tensor, attention_mask: Optional[Tensor], band_path_idx: Optional[Tensor] = None) -> Tensor:
        return self.head(self.encode_condition(idx, attention_mask, band_path_idx=band_path_idx))


def masked_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    loss = F.mse_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    loss = torch.abs(pred - target)
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _split_target(
    target: Tensor,
    band_shape: Optional[Tuple[int, int]],
    dos_dim: Optional[int],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if target is None or band_shape is None or dos_dim is None:
        return None, None
    band_dim = int(band_shape[0] * band_shape[1])
    bands = target[..., :band_dim].reshape(*target.shape[:-1], band_shape[0], band_shape[1])
    if int(dos_dim) <= 0:
        return bands, None
    dos = target[..., band_dim:band_dim + int(dos_dim)]
    return bands, dos


def masked_band_slope_loss(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    pred_slope = pred[..., 1:, :] - pred[..., :-1, :]
    target_slope = target[..., 1:, :] - target[..., :-1, :]
    slope_mask = None if mask is None else mask[..., 1:, :] * mask[..., :-1, :]
    return masked_mse(pred_slope, target_slope, slope_mask)


def masked_band_curvature_loss(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if pred.shape[-2] < 3:
        return pred.new_tensor(0.0)
    pred_curvature = pred[..., 2:, :] - 2.0 * pred[..., 1:-1, :] + pred[..., :-2, :]
    target_curvature = target[..., 2:, :] - 2.0 * target[..., 1:-1, :] + target[..., :-2, :]
    curvature_mask = None if mask is None else mask[..., 2:, :] * mask[..., 1:-1, :] * mask[..., :-2, :]
    return masked_mse(pred_curvature, target_curvature, curvature_mask)


def masked_dos_slope_loss(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    pred_slope = pred[..., 1:] - pred[..., :-1]
    target_slope = target[..., 1:] - target[..., :-1]
    slope_mask = None if mask is None else mask[..., 1:] * mask[..., :-1]
    return masked_mse(pred_slope, target_slope, slope_mask)


def _band_stat_tensor(stat: Optional[Tensor], band_shape: Tuple[int, int], ref: Tensor) -> Optional[Tensor]:
    if stat is None:
        return None
    band_dim = int(band_shape[0] * band_shape[1])
    stat = stat.to(device=ref.device, dtype=ref.dtype).reshape(-1)[:band_dim]
    if stat.numel() != band_dim:
        return None
    return stat.reshape(1, int(band_shape[0]), int(band_shape[1]))


def _denormalize_band(
    band: Tensor,
    band_shape: Tuple[int, int],
    target_mean: Optional[Tensor],
    target_std: Optional[Tensor],
) -> Tensor:
    mean = _band_stat_tensor(target_mean, band_shape, band)
    std = _band_stat_tensor(target_std, band_shape, band)
    if mean is None or std is None:
        return band
    return band * std.clamp_min(1e-12) + mean


def masked_band_fermi_window_loss(
    pred: Tensor,
    target: Tensor,
    target_physical: Tensor,
    mask: Optional[Tensor],
    window_ev: float,
) -> Tensor:
    if float(window_ev) <= 0.0:
        return pred.new_tensor(0.0)
    window_mask = (torch.abs(target_physical) <= float(window_ev)).to(dtype=pred.dtype)
    if mask is not None:
        window_mask = window_mask * mask.to(dtype=pred.dtype)
    return masked_mse(pred, target, window_mask)


def masked_band_gap_loss(
    pred_physical: Tensor,
    target_physical: Tensor,
    mask: Optional[Tensor] = None,
    gap_tol: float = 1e-3,
) -> Tensor:
    valid = torch.isfinite(pred_physical) & torch.isfinite(target_physical)
    if mask is not None:
        valid = valid & (mask > 0)

    occupied = valid & (target_physical <= 0.0)
    unoccupied = valid & (target_physical > 0.0)
    flat_occupied = occupied.reshape(occupied.shape[0], -1)
    flat_unoccupied = unoccupied.reshape(unoccupied.shape[0], -1)
    valid_sample = flat_occupied.any(dim=1) & flat_unoccupied.any(dim=1)
    if not torch.any(valid_sample):
        return pred_physical.new_tensor(0.0)

    large = torch.finfo(pred_physical.dtype).max / 8.0
    pred_vbm = torch.where(occupied, pred_physical, pred_physical.new_full((), -large)).reshape(pred_physical.shape[0], -1).amax(dim=1)
    target_vbm = torch.where(occupied, target_physical, target_physical.new_full((), -large)).reshape(target_physical.shape[0], -1).amax(dim=1)
    pred_cbm = torch.where(unoccupied, pred_physical, pred_physical.new_full((), large)).reshape(pred_physical.shape[0], -1).amin(dim=1)
    target_cbm = torch.where(unoccupied, target_physical, target_physical.new_full((), large)).reshape(target_physical.shape[0], -1).amin(dim=1)

    pred_gap = torch.relu(pred_cbm - pred_vbm)
    target_gap = torch.relu(target_cbm - target_vbm)
    target_gap = torch.where(target_gap <= float(gap_tol), target_gap.new_zeros(()), target_gap)
    return F.mse_loss(pred_gap[valid_sample], target_gap[valid_sample], reduction="mean")


def masked_reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    band_shape: Optional[Tuple[int, int]] = None,
    dos_dim: Optional[int] = None,
    band_weight: float = 1.0,
    dos_weight: float = 1.0,
    band_slope_weight: float = 0.05,
    dos_slope_weight: float = 0.05,
    band_curvature_weight: float = 0.0,
    band_fermi_window: float = 0.0,
    band_fermi_weight: float = 0.0,
    band_gap_weight: float = 0.0,
    band_gap_tol: float = 1e-3,
    target_mean: Optional[Tensor] = None,
    target_std: Optional[Tensor] = None,
) -> Tensor:
    if band_shape is None:
        return masked_mse(pred, target, mask)

    pred_band, pred_dos = _split_target(pred, band_shape, dos_dim)
    target_band, target_dos = _split_target(target, band_shape, dos_dim)
    if mask is None:
        mask_band, mask_dos = None, None
    else:
        mask_band, mask_dos = _split_target(mask, band_shape, dos_dim)

    band_loss = masked_mse(pred_band, target_band, mask_band)
    slope_loss = masked_band_slope_loss(pred_band, target_band, mask_band)
    curvature_loss = masked_band_curvature_loss(pred_band, target_band, mask_band)
    band_total = band_loss + float(band_slope_weight) * slope_loss + float(band_curvature_weight) * curvature_loss
    if float(band_fermi_weight) > 0.0 and float(band_fermi_window) > 0.0:
        target_band_physical = _denormalize_band(target_band, band_shape, target_mean, target_std)
        band_total = band_total + float(band_fermi_weight) * masked_band_fermi_window_loss(
            pred_band,
            target_band,
            target_band_physical,
            mask_band,
            band_fermi_window,
        )
    if float(band_gap_weight) > 0.0:
        pred_band_physical = _denormalize_band(pred_band, band_shape, target_mean, target_std)
        target_band_physical = _denormalize_band(target_band, band_shape, target_mean, target_std)
        band_total = band_total + float(band_gap_weight) * masked_band_gap_loss(
            pred_band_physical,
            target_band_physical,
            mask_band,
            gap_tol=band_gap_tol,
        )
    if dos_dim is None or int(dos_dim) <= 0 or pred_dos is None or target_dos is None:
        return float(band_weight) * band_total
    dos_loss = masked_mse(pred_dos, target_dos, mask_dos)
    dos_slope_loss = masked_dos_slope_loss(pred_dos, target_dos, mask_dos)
    return (
        float(band_weight) * band_total
        + float(dos_weight) * (dos_loss + float(dos_slope_weight) * dos_slope_loss)
    )
