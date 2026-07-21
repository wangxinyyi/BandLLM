import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ._tokenizer import CIFTokenizer


@dataclass
class ElectronicGridConfig:
    band_kpoints: int = 128
    band_count: int = 16
    band_axis: str = "distance"
    band_selection: str = "fermi"
    band_emin: Optional[float] = -10.0
    band_emax: Optional[float] = 10.0
    dos_points: int = 1024
    dos_emin: float = -10.0
    dos_emax: float = 10.0
    block_size: int = 2048


DEFAULT_TARGET_BAND_COUNT = 16
DEFAULT_BAND_EMIN = -10.0
DEFAULT_BAND_EMAX = 10.0


@dataclass
class ElectronicStructure:
    bands: Optional[np.ndarray] = None
    band_axis: Optional[np.ndarray] = None
    band_path: str = ""
    dos_energies: Optional[np.ndarray] = None
    dos_values: Optional[np.ndarray] = None
    efermi: float = 0.0
    source: str = ""


def read_cif(path: str) -> str:
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def tokenize_cif_to_block(cif: str, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    tokenizer = CIFTokenizer()
    tokens = tokenizer.tokenize_cif(cif)
    ids = np.asarray(tokenizer.encode(tokens), dtype=np.int64)
    if len(ids) > block_size:
        ids = ids[:block_size]

    attention_mask = np.zeros(block_size, dtype=np.float32)
    attention_mask[:len(ids)] = 1.0
    out = np.zeros(block_size, dtype=np.int64)
    out[:len(ids)] = ids
    return out, attention_mask


def _sum_spin_densities(densities: Dict[object, np.ndarray]) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32) for v in densities.values()]
    if not values:
        raise ValueError("DOS densities are empty")
    return np.sum(np.stack(values, axis=0), axis=0)


def _normalized_axis_from_distances(distances) -> Optional[np.ndarray]:
    if distances is None:
        return None
    axis = np.asarray(distances, dtype=np.float32)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        return None
    axis = axis - axis[0]
    span = float(axis[-1])
    if span <= 0.0:
        return None
    axis = axis / span
    if np.any(np.diff(axis) < -1e-5):
        return None
    return axis.astype(np.float32)


def _cartesian_kpoint_coords_from_pymatgen(band_structure) -> Optional[np.ndarray]:
    kpoints = getattr(band_structure, "kpoints", None)
    if not kpoints:
        return None

    lattice_rec = getattr(band_structure, "lattice_rec", None)
    lattice_matrix = None
    if lattice_rec is not None and getattr(lattice_rec, "matrix", None) is not None:
        lattice_matrix = np.asarray(lattice_rec.matrix, dtype=np.float32)

    coords = []
    for kpoint in kpoints:
        cart = getattr(kpoint, "cart_coords", None)
        if cart is not None:
            coords.append(np.asarray(cart, dtype=np.float32))
            continue

        frac = getattr(kpoint, "frac_coords", None)
        if frac is None:
            return None
        frac = np.asarray(frac, dtype=np.float32)
        if lattice_matrix is not None:
            coords.append(frac @ lattice_matrix)
        else:
            coords.append(frac)

    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _normalized_axis_from_coords(coords: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] != 3:
        return None
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    distances = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(steps).astype(np.float32)])
    return _normalized_axis_from_distances(distances)


def band_axis_from_pymatgen_band_structure(band_structure) -> Optional[np.ndarray]:
    axis = _normalized_axis_from_distances(getattr(band_structure, "distance", None))
    if axis is not None:
        return axis
    return _normalized_axis_from_coords(_cartesian_kpoint_coords_from_pymatgen(band_structure))


def band_path_signature_from_pymatgen(band_structure) -> str:
    branches = getattr(band_structure, "branches", None)
    if branches:
        names = []
        for branch in branches:
            name = branch.get("name", "") if isinstance(branch, dict) else getattr(branch, "name", "")
            if name:
                names.append(str(name).replace("\\Gamma", "G"))
        if names:
            return "|".join(names)

    labels_dict = getattr(band_structure, "labels_dict", None)
    if labels_dict:
        return "labels:" + "-".join(sorted(str(label).replace("\\Gamma", "G") for label in labels_dict.keys()))
    return "unknown"


def normalized_axis_from_kpoint_coords(kpoint_coords) -> Optional[np.ndarray]:
    return _normalized_axis_from_coords(np.asarray(kpoint_coords, dtype=np.float32))


def _select_band_indices(
    bands: np.ndarray,
    n_bands: int,
    selection: str,
    emin: Optional[float],
    emax: Optional[float],
) -> np.ndarray:
    if selection == "lowest":
        return np.arange(min(n_bands, bands.shape[1]), dtype=np.int64)
    if selection != "fermi":
        raise ValueError(f"unknown band selection mode: {selection}")

    scores = []
    for band_idx in range(bands.shape[1]):
        band = bands[:, band_idx]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        in_window = np.ones_like(finite, dtype=bool)
        if emin is not None:
            in_window &= finite >= float(emin)
        if emax is not None:
            in_window &= finite <= float(emax)
        if in_window.any():
            score = float(np.min(np.abs(finite[in_window])))
            window_penalty = 0
        else:
            score = float(np.min(np.abs(finite)))
            window_penalty = 1
        center = float(np.median(finite))
        scores.append((window_penalty, score, abs(center), band_idx))

    scores.sort()
    selected = sorted(item[-1] for item in scores[:n_bands])
    return np.asarray(selected, dtype=np.int64)


def electronic_structure_from_pymatgen(
    band_structure=None,
    dos=None,
    source: str = "",
) -> ElectronicStructure:
    efermi = 0.0
    bands = None
    band_axis = None
    band_path = ""
    dos_energies = None
    dos_values = None

    if band_structure is not None:
        efermi = float(getattr(band_structure, "efermi", 0.0) or 0.0)
        spin_bands = []
        for values in band_structure.bands.values():
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim != 2:
                continue
            spin_bands.append(arr.T)
        if spin_bands:
            bands = np.mean(np.stack(spin_bands, axis=0), axis=0)
            band_axis = band_axis_from_pymatgen_band_structure(band_structure)
            band_path = band_path_signature_from_pymatgen(band_structure)

    if dos is not None:
        efermi = float(getattr(dos, "efermi", efermi) or efermi)
        dos_energies = np.asarray(dos.energies, dtype=np.float32)
        dos_values = _sum_spin_densities(dos.densities)

    return ElectronicStructure(
        bands=bands,
        band_axis=band_axis,
        band_path=band_path,
        dos_energies=dos_energies,
        dos_values=dos_values,
        efermi=efermi,
        source=source,
    )


def load_from_vasprun(path: str) -> ElectronicStructure:
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as exc:
        raise ImportError("pymatgen is required to parse vasprun.xml") from exc

    vasprun = Vasprun(path, parse_projected_eigen=False, parse_potcar_file=False)
    efermi = float(vasprun.efermi or 0.0)

    bands = None
    band_axis = None
    if getattr(vasprun, "eigenvalues", None):
        spin_bands = []
        for values in vasprun.eigenvalues.values():
            arr = np.asarray(values, dtype=np.float32)
            spin_bands.append(arr[:, :, 0])
        bands = np.mean(np.stack(spin_bands, axis=0), axis=0)
        if getattr(vasprun, "actual_kpoints", None) is not None:
            band_axis = normalized_axis_from_kpoint_coords(vasprun.actual_kpoints)

    dos_energies = None
    dos_values = None
    if getattr(vasprun, "complete_dos", None) is not None:
        dos_energies = np.asarray(vasprun.complete_dos.energies, dtype=np.float32)
        dos_values = _sum_spin_densities(vasprun.complete_dos.densities)

    return ElectronicStructure(
        bands=bands,
        band_axis=band_axis,
        band_path="vasprun-kpoints" if band_axis is not None else "",
        dos_energies=dos_energies,
        dos_values=dos_values,
        efermi=efermi,
        source=path,
    )


def load_from_doscar(path: str) -> ElectronicStructure:
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 7:
        raise ValueError(f"DOSCAR is too short: {path}")

    header = lines[5].split()
    if len(header) < 4:
        raise ValueError(f"Could not parse DOSCAR grid header: {path}")

    nedos = int(float(header[2]))
    efermi = float(header[3])
    rows = []
    for line in lines[6:6 + nedos]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append([float(v) for v in parts])
    if not rows:
        raise ValueError(f"No DOS rows parsed from DOSCAR: {path}")

    arr = np.asarray(rows, dtype=np.float32)
    energies = arr[:, 0]
    if arr.shape[1] >= 4:
        densities = arr[:, 1] + arr[:, 2]
    else:
        densities = arr[:, 1]

    return ElectronicStructure(
        dos_energies=energies,
        dos_values=densities,
        efermi=efermi,
        source=path,
    )


def load_from_eigenval(path: str, efermi: float = 0.0) -> ElectronicStructure:
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 7:
        raise ValueError(f"EIGENVAL is too short: {path}")

    counts = lines[5].split()
    if len(counts) < 3:
        raise ValueError(f"Could not parse EIGENVAL counts: {path}")
    nkpoints = int(counts[-2])
    nbands = int(counts[-1])

    bands: List[List[float]] = []
    kpoint_coords: List[List[float]] = []
    i = 6
    for _ in range(nkpoints):
        if i >= len(lines):
            break
        kpoint_parts = lines[i].split()
        if len(kpoint_parts) >= 3:
            kpoint_coords.append([float(kpoint_parts[0]), float(kpoint_parts[1]), float(kpoint_parts[2])])
        i += 1  # k-point line
        energies = []
        for _ in range(nbands):
            if i >= len(lines):
                break
            parts = lines[i].split()
            if len(parts) >= 2:
                energies.append(float(parts[1]))
            i += 1
        if energies:
            bands.append(energies)

    if not bands:
        raise ValueError(f"No band rows parsed from EIGENVAL: {path}")

    return ElectronicStructure(
        bands=np.asarray(bands, dtype=np.float32),
        band_axis=normalized_axis_from_kpoint_coords(kpoint_coords) if len(kpoint_coords) == len(bands) else None,
        band_path="eigenval-kpoints" if len(kpoint_coords) == len(bands) else "",
        efermi=efermi,
        source=path,
    )


def load_electronic_structure(
    vasprun_path: str = "",
    eigenval_path: str = "",
    doscar_path: str = "",
    efermi: float = 0.0,
) -> ElectronicStructure:
    result = ElectronicStructure(efermi=efermi)

    if vasprun_path:
        result = load_from_vasprun(vasprun_path)

    if eigenval_path:
        eigen = load_from_eigenval(eigenval_path, efermi=result.efermi)
        result.bands = eigen.bands
        result.band_axis = eigen.band_axis
        result.band_path = eigen.band_path
        result.source = eigen.source

    if doscar_path:
        dos = load_from_doscar(doscar_path)
        result.dos_energies = dos.dos_energies
        result.dos_values = dos.dos_values
        result.efermi = dos.efermi
        result.source = dos.source

    if result.bands is None and result.dos_values is None:
        raise ValueError("No electronic structure data was provided")
    return result


def resample_bands(
    bands: np.ndarray,
    efermi: float,
    n_kpoints: int,
    n_bands: int,
    band_axis: Optional[np.ndarray] = None,
    selection: str = "fermi",
    emin: Optional[float] = -10.0,
    emax: Optional[float] = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    bands = np.asarray(bands, dtype=np.float32) - float(efermi)
    if bands.ndim != 2:
        raise ValueError(f"bands must be [n_kpoints, n_bands], got shape {bands.shape}")
    selected = _select_band_indices(bands, n_bands, selection, emin, emax)

    old_x = _normalized_axis_from_distances(band_axis)
    if old_x is None or len(old_x) != bands.shape[0]:
        old_x = np.linspace(0.0, 1.0, bands.shape[0], dtype=np.float32)
    unique_x, inverse = np.unique(old_x, return_inverse=True)
    if len(unique_x) != len(old_x):
        merged = np.zeros((len(unique_x), bands.shape[1]), dtype=np.float32)
        counts = np.zeros(len(unique_x), dtype=np.float32)
        for src, dst in enumerate(inverse):
            merged[dst] += bands[src]
            counts[dst] += 1.0
        bands = merged / counts[:, None]
        old_x = unique_x.astype(np.float32)

    new_x = np.linspace(0.0, 1.0, n_kpoints, dtype=np.float32)
    out = np.zeros((n_kpoints, n_bands), dtype=np.float32)
    mask = np.zeros((n_kpoints, n_bands), dtype=np.float32)

    for out_idx, band_idx in enumerate(selected):
        values = np.interp(new_x, old_x, bands[:, band_idx]).astype(np.float32)
        out[:, out_idx] = values
        valid = np.ones(n_kpoints, dtype=bool)
        if emin is not None:
            valid &= values >= float(emin)
        if emax is not None:
            valid &= values <= float(emax)
        mask[:, out_idx] = valid.astype(np.float32)
    return out, mask


def resample_dos(
    energies: np.ndarray,
    densities: np.ndarray,
    efermi: float,
    n_points: int,
    emin: float,
    emax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies = np.asarray(energies, dtype=np.float32) - float(efermi)
    densities = np.asarray(densities, dtype=np.float32)
    if energies.ndim != 1 or densities.ndim != 1:
        raise ValueError("DOS energies and densities must be one-dimensional")
    if len(energies) != len(densities):
        raise ValueError("DOS energy and density arrays have different lengths")

    order = np.argsort(energies)
    energies = energies[order]
    densities = densities[order]
    grid = np.linspace(emin, emax, n_points, dtype=np.float32)
    values = np.interp(grid, energies, densities, left=0.0, right=0.0).astype(np.float32)
    mask = ((grid >= energies[0]) & (grid <= energies[-1])).astype(np.float32)
    return grid, values, mask


def select_target_bands(
    bands: np.ndarray,
    band_mask: Optional[np.ndarray] = None,
    n_bands: int = DEFAULT_TARGET_BAND_COUNT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bands = np.asarray(bands, dtype=np.float32)
    if bands.ndim != 2:
        raise ValueError(f"bands must be [kpoints, bands], got shape {bands.shape}")

    if band_mask is None:
        band_mask = np.isfinite(bands).astype(np.float32)
    else:
        band_mask = np.asarray(band_mask, dtype=np.float32)
    valid = (band_mask > 0.0) & np.isfinite(bands)

    valence_scores = []
    conduction_scores = []
    fallback_scores = []
    for band_idx in range(bands.shape[1]):
        band_valid = valid[:, band_idx]
        if not np.any(band_valid):
            continue
        values = bands[band_valid, band_idx]
        median = float(np.median(values))
        fallback_scores.append((float(np.min(np.abs(values))), abs(median), band_idx))
        below = values[values <= 0.0]
        above = values[values >= 0.0]
        if below.size:
            valence_scores.append((float(abs(np.max(below))), abs(median), band_idx))
        if above.size:
            conduction_scores.append((float(abs(np.min(above))), abs(median), band_idx))

    valence_scores.sort(key=lambda item: (item[0], item[1], item[2]))
    conduction_scores.sort(key=lambda item: (item[0], item[1], item[2]))
    fallback_scores.sort(key=lambda item: (item[0], item[1], item[2]))

    selected = []
    seen = set()
    valence_target = int(n_bands) // 2
    conduction_target = int(n_bands) - valence_target
    for _, _, band_idx in valence_scores:
        if len(selected) >= valence_target:
            break
        selected.append(band_idx)
        seen.add(band_idx)
    for _, _, band_idx in conduction_scores:
        if len(selected) >= valence_target + conduction_target:
            break
        if band_idx in seen:
            continue
        selected.append(band_idx)
        seen.add(band_idx)
    for _, _, band_idx in fallback_scores:
        if len(selected) >= int(n_bands):
            break
        if band_idx in seen:
            continue
        selected.append(band_idx)
        seen.add(band_idx)

    selected.sort(key=lambda idx: float(np.median(bands[valid[:, idx], idx])) if np.any(valid[:, idx]) else float(idx))

    out = np.zeros((bands.shape[0], int(n_bands)), dtype=np.float32)
    out_mask = np.zeros_like(out)
    selected_array = np.full(int(n_bands), -1, dtype=np.int32)
    for out_idx, src_idx in enumerate(selected):
        out[:, out_idx] = np.where(valid[:, src_idx], bands[:, src_idx], 0.0).astype(np.float32)
        out_mask[:, out_idx] = valid[:, src_idx].astype(np.float32)
        selected_array[out_idx] = int(src_idx)
    return out, out_mask, selected_array


def target_from_components(
    bands: np.ndarray,
    band_mask: np.ndarray,
    dos: np.ndarray,
    dos_mask: np.ndarray,
    target_bands: Optional[np.ndarray] = None,
    target_band_mask: Optional[np.ndarray] = None,
    target_band_indices: Optional[np.ndarray] = None,
    target_band_count: int = DEFAULT_TARGET_BAND_COUNT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_bands is None or target_band_mask is None:
        target_bands, target_band_mask, target_band_indices = select_target_bands(
            bands,
            band_mask,
            n_bands=target_band_count,
        )
    if target_band_indices is None:
        target_band_indices = np.arange(target_bands.shape[1], dtype=np.int32)

    target = np.asarray(target_bands.reshape(-1), dtype=np.float32)
    target_mask = np.asarray(target_band_mask.reshape(-1), dtype=np.float32)
    return (
        np.asarray(target_bands, dtype=np.float32),
        np.asarray(target_band_mask, dtype=np.float32),
        np.asarray(target_band_indices, dtype=np.int32),
        target,
        target_mask,
    )


def vectorize_electronic_structure(
    es: ElectronicStructure,
    config: ElectronicGridConfig,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}

    if es.bands is not None:
        bands, band_mask = resample_bands(
            es.bands,
            es.efermi,
            config.band_kpoints,
            config.band_count,
            es.band_axis if config.band_axis == "distance" else None,
            config.band_selection,
            config.band_emin,
            config.band_emax,
        )
    else:
        bands = np.zeros((config.band_kpoints, config.band_count), dtype=np.float32)
        band_mask = np.zeros_like(bands)

    if es.dos_energies is not None and es.dos_values is not None:
        dos_grid, dos_values, dos_mask = resample_dos(
            es.dos_energies,
            es.dos_values,
            es.efermi,
            config.dos_points,
            config.dos_emin,
            config.dos_emax,
        )
    else:
        dos_grid = np.linspace(config.dos_emin, config.dos_emax, config.dos_points, dtype=np.float32)
        dos_values = np.zeros(config.dos_points, dtype=np.float32)
        dos_mask = np.zeros(config.dos_points, dtype=np.float32)

    arrays["bands"] = bands
    arrays["band_mask"] = band_mask
    arrays["band_t"] = np.linspace(0.0, 1.0, config.band_kpoints, dtype=np.float32)
    arrays["band_path"] = np.asarray(es.band_path or "unknown")
    arrays["dos_grid"] = dos_grid
    arrays["dos"] = dos_values
    arrays["dos_mask"] = dos_mask
    target_bands, target_band_mask, target_band_indices, target, target_mask = target_from_components(
        bands,
        band_mask,
        dos_values,
        dos_mask,
        target_band_count=config.band_count,
    )
    arrays["band16"] = target_bands
    arrays["band16_mask"] = target_band_mask
    arrays["band16_indices"] = target_band_indices
    arrays["target"] = target
    arrays["target_mask"] = target_mask
    return arrays



def transform_target_for_training(
    target: np.ndarray,
    band_shape: Optional[Tuple[int, int]],
    dos_dim: Optional[int],
) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32).copy()
    if band_shape is None or dos_dim is None or int(dos_dim) <= 0:
        return arr
    band_dim = int(band_shape[0] * band_shape[1])
    arr[..., band_dim:band_dim + dos_dim] = np.log1p(np.clip(arr[..., band_dim:band_dim + dos_dim], a_min=0.0, a_max=None))
    return arr


def inverse_target_transform(
    target: np.ndarray,
    band_shape: Optional[Tuple[int, int]],
    dos_dim: Optional[int],
    clamp_dos: bool = True,
) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32).copy()
    if band_shape is None or dos_dim is None or int(dos_dim) <= 0:
        return arr
    band_dim = int(band_shape[0] * band_shape[1])
    dos_log = arr[..., band_dim:band_dim + dos_dim]
    # Keep evaluation numerically stable even if the model predicts out-of-range log DOS values.
    dos_log = np.clip(dos_log, a_min=-20.0, a_max=20.0)
    dos = np.expm1(dos_log)
    if clamp_dos:
        dos = np.maximum(dos, 0.0)
    arr[..., band_dim:band_dim + dos_dim] = dos
    return arr.astype(np.float32)


def write_manifest(rows: List[Dict[str, str]], path: str) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_manifest_path(manifest_path: str, path: str) -> str:
    root = os.path.dirname(os.path.abspath(manifest_path))
    return path if os.path.isabs(path) else os.path.join(root, path)


def filter_manifest_rows(
    rows: List[Dict[str, str]],
    split: str = "all",
    band_path: str = "",
) -> List[Dict[str, str]]:
    result = rows
    if split != "all":
        result = [row for row in result if row.get("split", "") == split]
    if band_path:
        result = [row for row in result if (row.get("band_path", "") or "unknown") == band_path]
    return result


def infer_target_layout_from_manifest(
    manifest_path: str,
    rows: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    rows = rows if rows is not None else read_manifest(manifest_path)
    if not rows:
        return None, None
    sample_path = resolve_manifest_path(manifest_path, rows[0]["sample_path"])
    with np.load(sample_path) as data:
        if "band16" in data.files:
            return tuple(data["band16"].shape), 0
        if "bands" in data.files:
            band_count = min(DEFAULT_TARGET_BAND_COUNT, int(data["bands"].shape[1]))
            return (int(data["bands"].shape[0]), band_count), 0
        return None, None


def electronic_summary_from_arrays(
    bands: np.ndarray,
    band_mask: np.ndarray,
    dos: np.ndarray,
    dos_grid: np.ndarray,
    dos_mask: np.ndarray,
    gap_tol: float = 1e-3,
) -> Dict[str, float]:
    bands = np.asarray(bands, dtype=np.float32)
    band_mask = np.asarray(band_mask, dtype=np.float32)
    dos = np.asarray(dos, dtype=np.float32)
    dos_grid = np.asarray(dos_grid, dtype=np.float32)
    dos_mask = np.asarray(dos_mask, dtype=np.float32)

    valid_band_points = bands[band_mask > 0]
    if valid_band_points.size == 0:
        vbm = 0.0
        cbm = 0.0
        band_gap = 0.0
        is_metal = 0.0
    else:
        occupied = valid_band_points[valid_band_points <= 0.0]
        unoccupied = valid_band_points[valid_band_points > 0.0]
        vbm = float(occupied.max()) if occupied.size else 0.0
        cbm = float(unoccupied.min()) if unoccupied.size else 0.0
        band_gap = max(cbm - vbm, 0.0) if occupied.size and unoccupied.size else 0.0
        is_metal = 0.0
        if bands.ndim == 2 and band_mask.ndim == 2:
            for band_idx in range(bands.shape[1]):
                valid = band_mask[:, band_idx] > 0
                if not np.any(valid):
                    continue
                band = bands[valid, band_idx]
                if np.min(band) <= gap_tol and np.max(band) >= -gap_tol:
                    is_metal = 1.0
                    break
        if band_gap <= gap_tol:
            is_metal = 1.0

    valid_dos = dos_mask > 0
    if not np.any(valid_dos):
        dos_fermi = 0.0
        dos_int_1ev = 0.0
        dos_int_3ev = 0.0
    else:
        grid = dos_grid[valid_dos]
        values = dos[valid_dos]
        fermi_idx = int(np.argmin(np.abs(grid)))
        dos_fermi = float(values[fermi_idx])
        window_1 = np.abs(grid) <= 1.0
        window_3 = np.abs(grid) <= 3.0
        dos_int_1ev = float(np.trapz(values[window_1], grid[window_1])) if np.any(window_1) else 0.0
        dos_int_3ev = float(np.trapz(values[window_3], grid[window_3])) if np.any(window_3) else 0.0

    return {
        "band_gap": float(band_gap),
        "vbm": float(vbm),
        "cbm": float(cbm),
        "dos_fermi": float(dos_fermi),
        "dos_int_1ev": float(dos_int_1ev),
        "dos_int_3ev": float(dos_int_3ev),
        "is_metal": float(is_metal),
    }


class ElectronicStructureDataset(Dataset):
    def __init__(self, manifest_path: str, rows: Optional[List[Dict[str, str]]] = None):
        self.manifest_path = manifest_path
        self.rows = rows if rows is not None else read_manifest(manifest_path)
        self.root = os.path.dirname(os.path.abspath(manifest_path))

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        sample_path = self._resolve(row["sample_path"])
        with np.load(sample_path) as data:
            bands = data["bands"].astype(np.float32)
            band_mask = data["band_mask"].astype(np.float32)
            dos = data["dos"].astype(np.float32) if "dos" in data.files else np.zeros(0, dtype=np.float32)
            dos_mask = (
                data["dos_mask"].astype(np.float32)
                if "dos_mask" in data.files
                else np.zeros_like(dos, dtype=np.float32)
            )
            dos_grid = (
                data["dos_grid"].astype(np.float32)
                if "dos_grid" in data.files
                else np.zeros_like(dos, dtype=np.float32)
            )
            if "band16" in data.files and "band16_mask" in data.files:
                target_bands = data["band16"].astype(np.float32)
                target_band_mask = data["band16_mask"].astype(np.float32)
                target_band_indices = (
                    data["band16_indices"].astype(np.int32)
                    if "band16_indices" in data.files
                    else np.arange(target_bands.shape[1], dtype=np.int32)
                )
            else:
                target_bands = None
                target_band_mask = None
                target_band_indices = None
            target_bands, target_band_mask, target_band_indices, target, target_mask = target_from_components(
                bands,
                band_mask,
                dos,
                dos_mask,
                target_bands=target_bands,
                target_band_mask=target_band_mask,
                target_band_indices=target_band_indices,
            )
            band_path = row.get("band_path", "")
            if not band_path and "band_path" in data.files:
                value = data["band_path"]
                band_path = str(value.item() if getattr(value, "shape", ()) == () else value)
            label_type = row.get("label_type", "")
            if not label_type and "label_type" in data.files:
                value = data["label_type"]
                label_type = str(value.item() if getattr(value, "shape", ()) == () else value)
            label_type = label_type or "full"
            has_band_value = row.get("has_band", "")
            if has_band_value == "" and "has_band" in data.files:
                has_band_value = str(int(data["has_band"]))
            if has_band_value == "":
                has_band_value = "1" if float(np.asarray(data["band_mask"]).sum()) > 0.0 else "0"
            has_dos_value = row.get("has_dos", "")
            if has_dos_value == "" and "has_dos" in data.files:
                has_dos_value = str(int(data["has_dos"]))
            if has_dos_value == "":
                has_dos_value = "1" if float(np.asarray(dos_mask).sum()) > 0.0 else "0"
            return {
                "input_ids": torch.from_numpy(data["input_ids"].astype(np.int64)),
                "attention_mask": torch.from_numpy(data["attention_mask"].astype(np.float32)),
                "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
                "bands": torch.from_numpy(target_bands),
                "band_mask": torch.from_numpy(target_band_mask),
                "band_indices": torch.from_numpy(target_band_indices),
                "dos_grid": torch.from_numpy(dos_grid),
                "dos": torch.from_numpy(dos),
                "dos_mask": torch.from_numpy(dos_mask),
                "band_path": band_path or "unknown",
                "label_type": label_type,
                "has_band": torch.tensor(has_band_value in {"1", "true", "True"}, dtype=torch.bool),
                "has_dos": torch.tensor(has_dos_value in {"1", "true", "True"}, dtype=torch.bool),
                "split": row.get("split", ""),
            }
