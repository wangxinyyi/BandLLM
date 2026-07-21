import argparse
import csv
from collections import Counter
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
TOKENIZER_PATH = os.path.join(REPO_DIR, "crystallm", "_tokenizer.py")


def load_cif_tokenizer():
    spec = importlib.util.spec_from_file_location("_crystallm_tokenizer_standalone", TOKENIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CIFTokenizer


CIFTokenizer = load_cif_tokenizer()

DEFAULT_HAS_PROPS = ["bandstructure", "dos"]
LABEL_MODE_HAS_PROPS = {
    "full": ["bandstructure", "dos"],
    "band-only": ["bandstructure"],
    "dos-only": ["dos"],
}


@dataclass
class ElectronicGridConfig:
    band_kpoints: int = 128
    band_count: int = 64
    band_axis: str = "distance"
    band_selection: str = "fermi"
    band_emin: Optional[float] = -10.0
    band_emax: Optional[float] = 10.0
    dos_points: int = 1024
    dos_emin: float = -10.0
    dos_emax: float = 10.0
    block_size: int = 2048


def material_ids_from_file(path: str) -> List[str]:
    with open(path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_field(doc, field: str, default=None):
    if isinstance(doc, dict):
        return doc.get(field, default)
    return getattr(doc, field, default)


def load_has_props(values: List[str]):
    if not values:
        return []
    try:
        from emmet.core.summary import HasProps
    except Exception:
        return values

    by_name = {item.name.lower(): item for item in HasProps}
    by_value = {str(item.value).lower(): item for item in HasProps}
    result = []
    for value in values:
        key = value.lower()
        if key in by_name:
            result.append(by_name[key])
        elif key in by_value:
            result.append(by_value[key])
        else:
            known = ", ".join(sorted(by_value))
            raise ValueError(f"unknown --has-props value {value!r}; known values include: {known}")
    return result


def prop_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def doc_has_props(doc) -> set[str]:
    values = get_field(doc, "has_props", []) or []
    return {prop_value(value) for value in values}


def default_has_props_for_label_mode(label_mode: str) -> List[str]:
    return list(LABEL_MODE_HAS_PROPS[label_mode])


def doc_matches_label_mode(doc, label_mode: str) -> bool:
    props = doc_has_props(doc)
    if not props:
        return True
    has_band = "bandstructure" in props
    has_dos = "dos" in props
    if label_mode == "full":
        return has_band and has_dos
    if label_mode == "band-only":
        return has_band and not has_dos
    if label_mode == "dos-only":
        return has_dos and not has_band
    raise ValueError(f"unknown label_mode: {label_mode}")


def label_mode_flags(label_mode: str) -> tuple[bool, bool]:
    if label_mode == "full":
        return True, True
    if label_mode == "band-only":
        return True, False
    if label_mode == "dos-only":
        return False, True
    raise ValueError(f"unknown label_mode: {label_mode}")


def optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def tokenize_cif_to_block(cif: str, block_size: int):
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


def sum_spin_densities(densities):
    values = [np.asarray(v, dtype=np.float32) for v in densities.values()]
    if not values:
        raise ValueError("DOS densities are empty")
    return np.sum(np.stack(values, axis=0), axis=0)


def normalized_axis_from_distances(distances):
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


def cartesian_kpoint_coords_from_pymatgen(band_structure):
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


def normalized_axis_from_coords(coords):
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] != 3:
        return None
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    distances = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(steps).astype(np.float32)])
    return normalized_axis_from_distances(distances)


def band_axis_from_pymatgen_band_structure(band_structure):
    axis = normalized_axis_from_distances(getattr(band_structure, "distance", None))
    if axis is not None:
        return axis
    return normalized_axis_from_coords(cartesian_kpoint_coords_from_pymatgen(band_structure))


def band_path_signature_from_pymatgen(band_structure):
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


def select_band_indices(bands, n_bands, selection, emin, emax):
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


def electronic_structure_from_pymatgen(band_structure=None, dos=None):
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
            if arr.ndim == 2:
                spin_bands.append(arr.T)
        if spin_bands:
            bands = np.mean(np.stack(spin_bands, axis=0), axis=0)
            band_axis = band_axis_from_pymatgen_band_structure(band_structure)
            band_path = band_path_signature_from_pymatgen(band_structure)

    if dos is not None:
        efermi = float(getattr(dos, "efermi", efermi) or efermi)
        dos_energies = np.asarray(dos.energies, dtype=np.float32)
        dos_values = sum_spin_densities(dos.densities)

    return {
        "bands": bands,
        "band_axis": band_axis,
        "band_path": band_path,
        "dos_energies": dos_energies,
        "dos_values": dos_values,
        "efermi": efermi,
    }


def resample_bands(bands, efermi, n_kpoints, n_bands, band_axis=None, selection="fermi", emin=-10.0, emax=10.0):
    bands = np.asarray(bands, dtype=np.float32) - float(efermi)
    selected = select_band_indices(bands, n_bands, selection, emin, emax)
    old_x = normalized_axis_from_distances(band_axis)
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


def resample_dos(energies, densities, efermi, n_points, emin, emax):
    energies = np.asarray(energies, dtype=np.float32) - float(efermi)
    densities = np.asarray(densities, dtype=np.float32)
    order = np.argsort(energies)
    energies = energies[order]
    densities = densities[order]
    grid = np.linspace(emin, emax, n_points, dtype=np.float32)
    values = np.interp(grid, energies, densities, left=0.0, right=0.0).astype(np.float32)
    mask = ((grid >= energies[0]) & (grid <= energies[-1])).astype(np.float32)
    return grid, values, mask


def vectorize_electronic_structure(es, config):
    if es["bands"] is not None:
        bands, band_mask = resample_bands(
            es["bands"],
            es["efermi"],
            config.band_kpoints,
            config.band_count,
            es.get("band_axis") if config.band_axis == "distance" else None,
            config.band_selection,
            config.band_emin,
            config.band_emax,
        )
    else:
        bands = np.zeros((config.band_kpoints, config.band_count), dtype=np.float32)
        band_mask = np.zeros_like(bands)

    if es["dos_energies"] is not None and es["dos_values"] is not None:
        dos_grid, dos_values, dos_mask = resample_dos(
            es["dos_energies"],
            es["dos_values"],
            es["efermi"],
            config.dos_points,
            config.dos_emin,
            config.dos_emax,
        )
    else:
        dos_grid = np.linspace(config.dos_emin, config.dos_emax, config.dos_points, dtype=np.float32)
        dos_values = np.zeros(config.dos_points, dtype=np.float32)
        dos_mask = np.zeros(config.dos_points, dtype=np.float32)

    return {
        "bands": bands,
        "band_mask": band_mask,
        "band_t": np.linspace(0.0, 1.0, config.band_kpoints, dtype=np.float32),
        "band_path": np.asarray(es.get("band_path") or "unknown"),
        "dos_grid": dos_grid,
        "dos": dos_values,
        "dos_mask": dos_mask,
        "target": bands.reshape(-1).astype(np.float32),
        "target_mask": band_mask.reshape(-1).astype(np.float32),
    }


def write_manifest(rows, path):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_manifest_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(lines: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def as_plain_dict(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def get_indexed_field(doc, *keys, default=None):
    for key in keys:
        if isinstance(doc, dict) and key in doc:
            return doc[key]
        if not isinstance(doc, dict) and hasattr(doc, str(key)):
            return getattr(doc, str(key))
    return default


def query_raw_electronic_structure_doc(mpr, material_id: str, fields: List[str]) -> Dict:
    materials = getattr(mpr, "materials", None)
    rester = getattr(materials, "electronic_structure", None) if materials is not None else None
    if rester is None:
        rester = getattr(mpr, "electronic_structure", None)
    if rester is None or not hasattr(rester, "_query_resource_data"):
        raise RuntimeError("mp-api client does not expose electronic_structure._query_resource_data")

    docs = rester._query_resource_data(
        criteria={"material_ids": material_id},
        fields=fields,
        use_document_model=False,
    )
    if not docs:
        raise ValueError(f"No electronic structure metadata found for {material_id}")
    doc = as_plain_dict(docs[0])
    if not isinstance(doc, dict):
        raise TypeError(f"Expected raw electronic structure metadata dict for {material_id}, got {type(doc).__name__}")
    return doc


def band_task_id_from_doc(doc: Dict, material_id: str, path_type: str = "setyawan_curtarolo") -> str:
    band_data = as_plain_dict(get_field(doc, "bandstructure"))
    path_data = as_plain_dict(get_field(band_data, path_type)) if band_data else None
    task_id = get_field(path_data, "task_id") if path_data else None
    if not task_id:
        raise ValueError(f"No {path_type} band structure task_id found for {material_id}")
    return str(task_id)


def dos_task_id_from_doc(doc: Dict, material_id: str) -> str:
    dos_data = as_plain_dict(get_field(doc, "dos"))
    task_id = get_field(dos_data, "task_id") if dos_data else None
    if not task_id:
        total = as_plain_dict(get_field(dos_data, "total")) if dos_data else None
        spin_data = as_plain_dict(get_indexed_field(total, "1", 1, default=None)) if total else None
        task_id = get_field(spin_data, "task_id") if spin_data else None
    if not task_id:
        raise ValueError(f"No total DOS task_id found for {material_id}")
    return str(task_id)


def electronic_bandstructure_rester(mpr):
    materials = getattr(mpr, "materials", None)
    rester = getattr(materials, "electronic_structure_bandstructure", None) if materials is not None else None
    if rester is None:
        rester = getattr(mpr, "electronic_structure_bandstructure", None)
    if rester is None:
        raise RuntimeError("mp-api client does not expose electronic_structure_bandstructure")
    return rester


def electronic_dos_rester(mpr):
    materials = getattr(mpr, "materials", None)
    rester = getattr(materials, "electronic_structure_dos", None) if materials is not None else None
    if rester is None:
        rester = getattr(mpr, "electronic_structure_dos", None)
    if rester is None:
        raise RuntimeError("mp-api client does not expose electronic_structure_dos")
    return rester


def get_bandstructure_by_material_id_raw(mpr, material_id: str):
    doc = query_raw_electronic_structure_doc(mpr, material_id, fields=["bandstructure"])
    task_id = band_task_id_from_doc(doc, material_id)
    return electronic_bandstructure_rester(mpr).get_bandstructure_from_task_id(task_id)


def get_dos_by_material_id_raw(mpr, material_id: str):
    doc = query_raw_electronic_structure_doc(mpr, material_id, fields=["dos"])
    task_id = dos_task_id_from_doc(doc, material_id)
    return electronic_dos_rester(mpr).get_dos_from_task_id(task_id)


def resolve_material_ids(mpr, args) -> List[str]:
    material_ids = list(args.material_ids or [])
    if args.material_ids_file:
        material_ids.extend(material_ids_from_file(args.material_ids_file))
    if material_ids:
        deduped = list(dict.fromkeys(material_ids))
        return deduped[:args.max_materials] if args.max_materials > 0 else deduped

    chunk_size = max(1, args.query_chunk_size)
    search_kwargs: Dict[str, object] = {
        "fields": ["material_id", "has_props"],
        "chunk_size": chunk_size,
    }
    if args.max_materials > 0:
        search_kwargs["num_chunks"] = max(1, math.ceil(args.max_materials / chunk_size))

    has_props = load_has_props(args.has_props)
    if has_props:
        search_kwargs["has_props"] = has_props

    if args.mp_all:
        active_filters = [args.chemsys, args.elements, args.formula, args.is_metal is not None, args.band_gap_min is not None, args.band_gap_max is not None]
        if any(active_filters):
            print("--mp-all was set; ignoring chemsys/elements/formula/is-metal/band-gap filters for id resolution")
    else:
        if args.chemsys:
            search_kwargs["chemsys"] = args.chemsys
        if args.elements:
            search_kwargs["elements"] = args.elements
        if args.formula:
            search_kwargs["formula"] = args.formula
        if args.is_metal is not None:
            search_kwargs["is_metal"] = args.is_metal
        if args.band_gap_min is not None or args.band_gap_max is not None:
            search_kwargs["band_gap"] = (
                args.band_gap_min if args.band_gap_min is not None else 0.0,
                args.band_gap_max if args.band_gap_max is not None else 100.0,
            )

    try:
        docs = mpr.materials.summary.search(**search_kwargs)
    except Exception as exc:
        if "has_props" not in search_kwargs:
            raise
        fallback_kwargs = dict(search_kwargs)
        fallback_kwargs["has_props"] = [str(value.value) if hasattr(value, "value") else str(value) for value in has_props]
        print(f"summary search with HasProps enums failed ({type(exc).__name__}: {exc}); retrying with string has_props")
        docs = mpr.materials.summary.search(**fallback_kwargs)

    resolved = []
    for doc in docs:
        if not doc_matches_label_mode(doc, args.label_mode):
            continue
        material_id = get_field(doc, "material_id")
        if material_id:
            resolved.append(str(material_id))
        if args.max_materials > 0 and len(resolved) >= args.max_materials:
            break
    return list(dict.fromkeys(resolved))


def structure_matches_filters(structure, args) -> bool:
    if args.mp_all:
        return True
    if not args.elements:
        return True
    present = {el.symbol for el in structure.composition.elements}
    return all(element in present for element in args.elements)


def main():
    parser = argparse.ArgumentParser(description="Download CIF, band structure, and DOS samples from Materials Project.")
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY", ""), help="Materials Project API key. Defaults to MP_API_KEY.")
    parser.add_argument("--api-key-file", default="", help="Path to a file containing the Materials Project API key.")
    parser.add_argument(
        "--use-document-model",
        action="store_true",
        help="Use mp-api/emmet pydantic document models. Disabled by default to support newer alphanumeric MPIDs on older clients.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mp-all", action="store_true", help="Resolve all MP materials with the requested --has-props, ignoring chemistry filters.")
    parser.add_argument(
        "--label-mode",
        choices=["full", "band-only", "dos-only"],
        default="full",
        help="Which electronic labels to require and download for each sample.",
    )
    parser.add_argument(
        "--has-props",
        nargs="*",
        default=None,
        help="Override summary has_props filters used for id resolution. Defaults depend on --label-mode.",
    )
    parser.add_argument("--material-ids", nargs="*", default=[])
    parser.add_argument("--material-ids-file", default="")
    parser.add_argument("--chemsys", default="", help="Example: Si-O or Li-Fe-P.")
    parser.add_argument("--elements", nargs="*", default=[])
    parser.add_argument("--formula", default="", help="Example: SiO2 or ABO3.")
    parser.add_argument("--is-metal", dest="is_metal", action="store_true")
    parser.add_argument("--is-nonmetal", dest="is_metal", action="store_false")
    parser.set_defaults(is_metal=None)
    parser.add_argument("--band-gap-min", type=float, default=None)
    parser.add_argument("--band-gap-max", type=float, default=None)
    parser.add_argument("--max-materials", type=int, default=25, help="Maximum material ids to resolve. Use 0 for all ids returned by the query.")
    parser.add_argument("--query-chunk-size", type=int, default=1000, help="Number of material ids to request per summary.search chunk.")
    parser.add_argument("--band-kpoints", type=int, default=128)
    parser.add_argument("--band-count", type=int, default=64)
    parser.add_argument("--band-axis", choices=["distance", "index"], default="distance", help="Use normalized reciprocal-space path distance or raw point index for band interpolation.")
    parser.add_argument("--band-selection", choices=["fermi", "lowest"], default="fermi", help="Select bands closest to the Fermi level or the lowest bands.")
    parser.add_argument("--band-emin", type=optional_float, default=-10.0, help="Minimum E-E_F used for the band target mask. Use 'none' to disable.")
    parser.add_argument("--band-emax", type=optional_float, default=10.0, help="Maximum E-E_F used for the band target mask. Use 'none' to disable.")
    parser.add_argument("--dos-points", type=int, default=1024)
    parser.add_argument("--dos-emin", type=float, default=-10.0)
    parser.add_argument("--dos-emax", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--errors-csv", default="", help="Optional CSV path for failed material ids and errors.")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing manifest/errors and resolved material id list.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing state files and start a fresh download in the output directory.")
    parser.set_defaults(resume=True)
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, retry material ids already present in errors.csv.")
    parser.add_argument("--save-every", type=int, default=20, help="Persist manifest/errors every N processed material ids.")
    parser.add_argument("--resolve-only", action="store_true", help="Resolve and save material ids, then exit before downloading structures/bandstructures/DOS.")
    args = parser.parse_args()

    if args.has_props is None:
        args.has_props = default_has_props_for_label_mode(args.label_mode)
    need_band, need_dos = label_mode_flags(args.label_mode)

    if not args.api_key and args.api_key_file:
        with open(args.api_key_file, "rt", encoding="utf-8") as f:
            args.api_key = f.read().strip()

    if not args.api_key:
        raise ValueError("Provide --api-key or set MP_API_KEY.")

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError("Install mp-api first: pip install mp-api") from exc

    os.makedirs(args.out_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, "samples")
    cif_dir = os.path.join(args.out_dir, "cifs")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(cif_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    errors_csv = args.errors_csv or os.path.join(args.out_dir, "errors.csv")
    resolved_ids_path = os.path.join(args.out_dir, "resolved_material_ids.txt")

    grid_config = ElectronicGridConfig(
        band_kpoints=args.band_kpoints,
        band_count=args.band_count,
        band_axis=args.band_axis,
        band_selection=args.band_selection,
        band_emin=args.band_emin,
        band_emax=args.band_emax,
        dos_points=args.dos_points,
        dos_emin=args.dos_emin,
        dos_emax=args.dos_emax,
        block_size=args.block_size,
    )

    out_rows = read_manifest_rows(manifest_path) if args.resume else []
    error_rows = read_manifest_rows(errors_csv) if args.resume else []
    error_types = Counter()
    for row in error_rows:
        error_name = row.get("error_type", "unknown")
        error_types[error_name] += 1
    tokenizer = CIFTokenizer()
    vocab_size = len(tokenizer.token_to_id)
    completed_ids = {row["id"] for row in out_rows if row.get("id")}
    failed_ids = {row["id"] for row in error_rows if row.get("id")}

    with MPRester(args.api_key, use_document_model=args.use_document_model, mute_progress_bars=True) as mpr:
        if args.resume and os.path.exists(resolved_ids_path):
            material_ids = read_lines(resolved_ids_path)
            print(f"loaded {len(material_ids)} resolved material ids from {resolved_ids_path}")
        else:
            material_ids = resolve_material_ids(mpr, args)
            write_lines(material_ids, resolved_ids_path)
            print(f"resolved and saved {len(material_ids)} material ids to {resolved_ids_path}")

        if args.resolve_only:
            print(f"resolve-only mode: exiting before downloading samples")
            return

        skipped_existing = 0
        skipped_failed = 0
        if args.resume:
            if completed_ids:
                skipped_existing = sum(1 for mid in material_ids if mid in completed_ids)
            if failed_ids and not args.retry_failed:
                skipped_failed = sum(1 for mid in material_ids if mid in failed_ids and mid not in completed_ids)

        print(
            f"downloading up to {len(material_ids)} Materials Project electronic-structure samples "
            f"(label_mode={args.label_mode}, has_band={int(need_band)}, has_dos={int(need_dos)})"
        )
        if args.resume:
            print(f"resume mode: existing completed={len(completed_ids)}, existing failed={len(failed_ids)}, skip_completed={skipped_existing}, skip_failed={skipped_failed}, retry_failed={args.retry_failed}")

        processed_since_save = 0
        for material_id in tqdm(material_ids, desc="downloading MP samples"):
            if args.resume and material_id in completed_ids:
                continue
            if args.resume and (not args.retry_failed) and material_id in failed_ids:
                continue
            try:
                structure = mpr.get_structure_by_material_id(material_id)
                if not structure_matches_filters(structure, args):
                    continue
                band_structure = get_bandstructure_by_material_id_raw(mpr, material_id) if need_band else None
                dos = get_dos_by_material_id_raw(mpr, material_id) if need_dos else None

                cif = structure.to(fmt="cif")
                cif_path = os.path.join(cif_dir, f"{material_id}.cif")
                with open(cif_path, "wt", encoding="utf-8") as f:
                    f.write(cif)

                input_ids, attention_mask = tokenize_cif_to_block(cif, grid_config.block_size)
                electronic = electronic_structure_from_pymatgen(
                    band_structure=band_structure,
                    dos=dos,
                )
                arrays = vectorize_electronic_structure(electronic, grid_config)

                sample_path = os.path.join(sample_dir, f"{material_id}.npz")
                np.savez_compressed(
                    sample_path,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label_type=np.asarray(args.label_mode.replace("-", "_")),
                    has_band=np.asarray(1 if need_band else 0, dtype=np.int8),
                    has_dos=np.asarray(1 if need_dos else 0, dtype=np.int8),
                    **arrays,
                )
                out_rows.append({
                    "id": material_id,
                    "sample_path": os.path.relpath(sample_path, args.out_dir),
                    "cif_path": os.path.relpath(cif_path, args.out_dir),
                    "band_path": electronic.get("band_path") or "unknown",
                    "label_type": args.label_mode.replace("-", "_"),
                    "has_band": "1" if need_band else "0",
                    "has_dos": "1" if need_dos else "0",
                    "source": f"materials-project:{material_id}",
                    "vocab_size": str(vocab_size),
                })
                completed_ids.add(material_id)
                if material_id in failed_ids:
                    failed_ids.remove(material_id)
                    error_rows = [row for row in error_rows if row.get("id") != material_id]
            except Exception as exc:
                error_name = type(exc).__name__
                error_types[error_name] += 1
                error_rows = [row for row in error_rows if row.get("id") != material_id]
                error_rows.append({
                    "id": material_id,
                    "error_type": error_name,
                    "error": str(exc).replace("\n", " ")[:500],
                })
                failed_ids.add(material_id)
                if not args.skip_errors:
                    raise
                print(f"skipping {material_id}: {exc}")
            processed_since_save += 1
            if processed_since_save >= max(1, args.save_every):
                write_manifest(out_rows, manifest_path)
                if error_rows:
                    write_manifest(error_rows, errors_csv)
                processed_since_save = 0

    write_manifest(out_rows, manifest_path)
    if error_rows:
        write_manifest(error_rows, errors_csv)
    print(f"wrote {len(out_rows)} usable samples to {args.out_dir}")
    if error_rows:
        print(f"failed samples: {len(error_rows)}")
        print(f"error summary: {dict(error_types)}")
        print(f"wrote errors to {errors_csv}")


if __name__ == "__main__":
    main()
