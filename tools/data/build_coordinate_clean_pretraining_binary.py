#!/usr/bin/env python3
"""Build a pretraining binary excluding LoTSS benchmark sky positions.

This is a non-destructive post-processing step. It reads the WCS footprints of
the original LoTSS pretraining FITS tiles, identifies tiles containing a source
position from the downstream LoTSS DR2 benchmark, maps those tiles to the
processed tensor indices produced by ``utils/fits_convert.py``, and writes a new
binary without the matching tensors. The original FITS files, tensors, and
binary are not modified.

Configuration is intentionally kept in the block below. Run from the repository
root after tensor conversion and normalization:

    python tools/data/build_coordinate_clean_pretraining_binary.py
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRETRAINING_FITS_DIR = Path("/mnt/large_volume/adema02/Datasets/LoTSS512")
PROCESSED_TENSOR_DIR = Path("/mnt/large_volume/adema02/Datasets/ssl/LoTSS512_pt_processed")
BENCHMARK_DIR = PROJECT_ROOT / "data" / "lotss_dr2_horton_hires_cutouts"

ALL_PROCESSED_DIRS = [
    Path("/mnt/large_volume/adema02/Datasets/ssl/MeerKAT512_pt_processed"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/ASKAP512_pt_processed"),
    PROCESSED_TENSOR_DIR,
    Path("/mnt/large_volume/adema02/Datasets/ssl/SKA512_pt_processed"),
]
OUTPUT_BINARY = Path("/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_v5_coord_clean.bin")
OUTPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/ssl/coordinate_exclusion")

WORKERS = 32
FOOTPRINT_PADDING_PIXELS = 0.0
EXCLUDE_UNASSESSED_TILES = True
BUILD_BINARY = True
BINARY_BLOCK_SIZE = 8192
BINARY_SEED = 12345
BINARY_WRITERS = 16


_SOURCE_NAME_PATTERN = re.compile(
    r"ILTJ(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2}(?:\.\d+)?)"
    r"(?P<sign>[+-])(?P<degree>\d{2})(?P<arcminute>\d{2})(?P<arcsecond>\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)
_REFERENCE_XYZ: np.ndarray | None = None
_REFERENCE_TREE: cKDTree | None = None
_REFERENCE_RA: np.ndarray | None = None
_REFERENCE_DEC: np.ndarray | None = None
_REFERENCE_NAMES: list[str] | None = None
_LOTSS_INITIAL_KEYS = ("fri", "frii", "hybrid", "spiral", "relaxed")
_LOTSS_INITIAL_CODES = {"INIT-1", "INIT-2", "INIT-3", "INIT-4", "INIT-5"}


def parse_lotss_source_name(value: str) -> tuple[float, float]:
    match = _SOURCE_NAME_PATTERN.search(value)
    if match is None:
        raise ValueError(f"Could not parse a LoTSS source coordinate from {value!r}")
    ra = 15.0 * (
        int(match.group("hour"))
        + int(match.group("minute")) / 60.0
        + float(match.group("second")) / 3600.0
    )
    dec = (
        int(match.group("degree"))
        + int(match.group("arcminute")) / 60.0
        + float(match.group("arcsecond")) / 3600.0
    )
    if match.group("sign") == "-":
        dec = -dec
    return ra, dec


def load_benchmark_coordinates(directory: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records: dict[str, tuple[float, float]] = {}
    for sidecar in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(sidecar.read_text())
        except Exception:
            continue
        raw_flags = payload.get("raw_flags")
        if isinstance(raw_flags, dict):
            active_labels = sum(int(raw_flags.get(key, 0) or 0) == 1 for key in _LOTSS_INITIAL_KEYS)
        else:
            labels = payload.get("labels")
            initial = labels.get("initial") if isinstance(labels, dict) else None
            active_labels = sum(
                isinstance(entry, dict) and entry.get("code") in _LOTSS_INITIAL_CODES
                for entry in (initial if isinstance(initial, list) else [])
            )
        if active_labels != 1 or not sidecar.with_suffix(".fits").is_file():
            continue
        source_name = str(payload.get("source_name") or sidecar.stem.split("_s", 1)[0])
        try:
            records[source_name] = parse_lotss_source_name(source_name)
        except ValueError:
            continue
    if not records:
        raise RuntimeError(f"No retained LoTSS benchmark coordinates found in {directory}")
    names = sorted(records)
    ra = np.asarray([records[name][0] for name in names], dtype=np.float64)
    dec = np.asarray([records[name][1] for name in names], dtype=np.float64)
    return ra, dec, names


def coordinates_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def angular_radius_to_chord(radius_deg: float) -> float:
    return 2.0 * math.sin(math.radians(radius_deg) / 2.0)


def _initialize_worker(ra: np.ndarray, dec: np.ndarray, names: list[str]) -> None:
    global _REFERENCE_XYZ, _REFERENCE_TREE, _REFERENCE_RA, _REFERENCE_DEC, _REFERENCE_NAMES
    _REFERENCE_RA = ra
    _REFERENCE_DEC = dec
    _REFERENCE_NAMES = names
    _REFERENCE_XYZ = coordinates_to_unit_vectors(ra, dec)
    _REFERENCE_TREE = cKDTree(_REFERENCE_XYZ)


def _image_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if int(hdu.header.get("NAXIS", 0) or 0) >= 2:
            return hdu
    raise ValueError("no two-dimensional image HDU")


def inspect_tile(task: tuple[int, str, str]) -> dict[str, Any]:
    index, fits_path_text, tensor_path_text = task
    fits_path = Path(fits_path_text)
    try:
        with fits.open(fits_path, memmap=True, do_not_scale_image_data=True) as hdul:
            hdu = _image_hdu(hdul)
            width = int(hdu.header.get("NAXIS1", 0) or 0)
            height = int(hdu.header.get("NAXIS2", 0) or 0)
            if width <= 0 or height <= 0:
                raise ValueError("invalid image dimensions")
            wcs = WCS(hdu.header).celestial
            if not wcs.has_celestial:
                raise ValueError("missing celestial WCS")

            center_x = (width - 1.0) / 2.0
            center_y = (height - 1.0) / 2.0
            center = wcs.pixel_to_world(center_x, center_y)
            corners = wcs.pixel_to_world(
                np.asarray([0.0, width - 1.0, width - 1.0, 0.0]),
                np.asarray([0.0, 0.0, height - 1.0, height - 1.0]),
            )
            radius_deg = float(np.max(center.separation(corners).degree))

            center_xyz = coordinates_to_unit_vectors(
                np.asarray([float(center.ra.degree)]),
                np.asarray([float(center.dec.degree)]),
            )[0]
            assert _REFERENCE_TREE is not None
            candidates = _REFERENCE_TREE.query_ball_point(
                center_xyz,
                angular_radius_to_chord(radius_deg),
            )
            if not candidates:
                return {"status": "retained", "index": index}

            assert _REFERENCE_RA is not None and _REFERENCE_DEC is not None and _REFERENCE_NAMES is not None
            candidate_coords = SkyCoord(
                ra=_REFERENCE_RA[candidates],
                dec=_REFERENCE_DEC[candidates],
                unit="deg",
                frame="icrs",
            )
            x_pixels, y_pixels = wcs.world_to_pixel(candidate_coords)
            padding = float(FOOTPRINT_PADDING_PIXELS)
            inside = (
                np.isfinite(x_pixels)
                & np.isfinite(y_pixels)
                & (x_pixels >= -padding)
                & (x_pixels <= width - 1.0 + padding)
                & (y_pixels >= -padding)
                & (y_pixels <= height - 1.0 + padding)
            )
            if not np.any(inside):
                return {"status": "retained", "index": index}

            inside_positions = np.flatnonzero(inside)
            separations = center.separation(candidate_coords[inside_positions]).arcsec
            nearest_position = int(inside_positions[int(np.argmin(separations))])
            reference_index = int(candidates[nearest_position])
            return {
                "status": "excluded",
                "index": index,
                "fits_path": str(fits_path),
                "tensor_path": tensor_path_text,
                "benchmark_source": _REFERENCE_NAMES[reference_index],
                "benchmark_ra_deg": float(_REFERENCE_RA[reference_index]),
                "benchmark_dec_deg": float(_REFERENCE_DEC[reference_index]),
                "tile_center_ra_deg": float(center.ra.degree),
                "tile_center_dec_deg": float(center.dec.degree),
                "center_separation_arcsec": float(separations.min()),
                "tensor_exists": Path(tensor_path_text).is_file(),
            }
    except Exception as error:
        return {
            "status": "unassessed",
            "index": index,
            "fits_path": str(fits_path),
            "tensor_path": tensor_path_text,
            "reason": f"{type(error).__name__}: {error}",
        }


def scan_pretraining_tiles(
    fits_directory: Path,
    tensor_directory: Path,
    benchmark_ra: np.ndarray,
    benchmark_dec: np.ndarray,
    benchmark_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    fits_paths = sorted(fits_directory.rglob("*.fits"))
    if not fits_paths:
        raise RuntimeError(f"No pretraining FITS files found in {fits_directory}")
    tasks = [
        (index, str(path), str(tensor_directory / f"{index:08d}.pt"))
        for index, path in enumerate(fits_paths)
    ]
    excluded: list[dict[str, Any]] = []
    unassessed: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with context.Pool(
        processes=max(1, int(WORKERS)),
        initializer=_initialize_worker,
        initargs=(benchmark_ra, benchmark_dec, benchmark_names),
    ) as pool:
        iterator = pool.imap_unordered(inspect_tile, tasks, chunksize=128)
        for result in tqdm(iterator, total=len(tasks), desc="Checking LoTSS tile footprints", unit="tile"):
            if result["status"] == "excluded":
                excluded.append(result)
            elif result["status"] == "unassessed":
                unassessed.append(result)
    excluded.sort(key=lambda item: int(item["index"]))
    unassessed.sort(key=lambda item: int(item["index"]))
    return excluded, unassessed, len(tasks)


def write_reports(
    excluded: list[dict[str, Any]],
    unassessed: list[dict[str, Any]],
    scanned_count: int,
    benchmark_count: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    match_fields = [
        "index",
        "fits_path",
        "tensor_path",
        "tensor_exists",
        "benchmark_source",
        "benchmark_ra_deg",
        "benchmark_dec_deg",
        "tile_center_ra_deg",
        "tile_center_dec_deg",
        "center_separation_arcsec",
    ]
    with (OUTPUT_DIR / "coordinate_exclusion_matches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=match_fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in match_fields} for record in excluded)
    (OUTPUT_DIR / "coordinate_exclusion_unassessed.txt").write_text(
        "".join(f"{item['fits_path']} | {item['reason']}\n" for item in unassessed)
    )
    manifest = {
        "scope": "LoTSS pretraining tiles versus all LoTSS DR2 benchmark positions",
        "pretraining_fits_directory": str(PRETRAINING_FITS_DIR),
        "processed_tensor_directory": str(PROCESSED_TENSOR_DIR),
        "benchmark_directory": str(BENCHMARK_DIR),
        "benchmark_positions": benchmark_count,
        "pretraining_tiles_scanned": scanned_count,
        "tiles_excluded": len(excluded),
        "excluded_tensors_present": sum(bool(item["tensor_exists"]) for item in excluded),
        "tiles_unassessed": len(unassessed),
        "unassessed_tensors_excluded": (
            sum(Path(item["tensor_path"]).is_file() for item in unassessed) if EXCLUDE_UNASSESSED_TILES else 0
        ),
        "footprint_padding_pixels": FOOTPRINT_PADDING_PIXELS,
        "exclude_unassessed_tiles": EXCLUDE_UNASSESSED_TILES,
        "output_binary": str(OUTPUT_BINARY) if BUILD_BINARY else None,
        "excluded_tensor_paths": [
            *[item["tensor_path"] for item in excluded if item["tensor_exists"]],
            *(
                [item["tensor_path"] for item in unassessed if Path(item["tensor_path"]).is_file()]
                if EXCLUDE_UNASSESSED_TILES
                else []
            ),
        ],
    }
    (OUTPUT_DIR / "coordinate_exclusion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    benchmark_ra, benchmark_dec, benchmark_names = load_benchmark_coordinates(BENCHMARK_DIR)
    print(f"Loaded {len(benchmark_names):,} unique LoTSS benchmark positions.")
    excluded, unassessed, scanned_count = scan_pretraining_tiles(
        PRETRAINING_FITS_DIR,
        PROCESSED_TENSOR_DIR,
        benchmark_ra,
        benchmark_dec,
        benchmark_names,
    )
    write_reports(excluded, unassessed, scanned_count, len(benchmark_names))
    excluded_paths = {record["tensor_path"] for record in excluded if record["tensor_exists"]}
    if EXCLUDE_UNASSESSED_TILES:
        excluded_paths.update(
            record["tensor_path"] for record in unassessed if Path(record["tensor_path"]).is_file()
        )
    print(
        f"Scanned {scanned_count:,} LoTSS pretraining tiles: excluded {len(excluded):,}, "
        f"could not assess {len(unassessed):,}, and omitted {len(excluded_paths):,} processed tensors."
    )
    print(f"Reports written to {OUTPUT_DIR}")

    if BUILD_BINARY:
        from utils.fits_convert import write_big_binary_from_processed_dirs_parallel

        write_big_binary_from_processed_dirs_parallel(
            ALL_PROCESSED_DIRS,
            OUTPUT_BINARY,
            block_size=BINARY_BLOCK_SIZE,
            seed=BINARY_SEED,
            workers=BINARY_WRITERS,
            span_len=BINARY_BLOCK_SIZE * 2,
            flush_every=0,
            excluded_paths=excluded_paths,
        )


if __name__ == "__main__":
    main()
