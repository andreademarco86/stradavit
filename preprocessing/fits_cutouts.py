"""
fits_cutouts.py

Tile a directory of FITS images into 512x512 cutouts with no post-processing.
- Non-overlapping tiles (stride == tile_size)
- Drop any tile with NaN/Inf or all-zeros
- Store all cutouts into a single output root, bucketed into subfolders
  with at most --max-per-dir files per bucket.
- Preserve raw array values (no normalization; no rescaling by us).
  (Astropy's default BSCALE/BZERO application is left on; see --no-scale.)
- Supports 2D images and N-D cubes (iterates over leading axes, using last 2 as YX)

"""
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional
import hashlib

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from tqdm import tqdm

import errno
import time
import random

# Silence harmless astropy memmap warnings
warnings.simplefilter("ignore", category=AstropyUserWarning)


def select_plane(data: np.ndarray, plane_index: int) -> Tuple[np.ndarray, int]:
    """
    Select a single 2D plane from an image/cube, treating the last two dims as (Y, X).
    If data is 2D, returns (data, 0). If data is >=3D, flattens leading dims and selects
    the plane at `plane_index` (0-based). Returns (plane, flat_index_selected).
    Raises ValueError if plane_index is out of range or data.ndim < 2.
    """
    if data.ndim < 2:
        raise ValueError("Data has fewer than 2 dimensions")
    if data.ndim == 2:
        return data, 0
    lead_shape = data.shape[:-2]
    total = int(np.prod(lead_shape))
    if plane_index < 0 or plane_index >= total:
        raise ValueError(f"plane_index {plane_index} out of range for lead_shape {lead_shape} (total {total})")
    idx = np.unravel_index(plane_index, lead_shape)
    plane = data[idx]
    return plane, plane_index


def list_fits_files(root: Path) -> Iterable[Path]:
    exts = {".fits", ".fit", ".fts", ".fz"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def valid_tile(tile: np.ndarray) -> bool:
    if np.isnan(tile).any():
        return False
    if np.isinf(tile).any():
        return False
    # all-zeros check; matches dtype exactly (no tolerance)
    if np.all(tile == 0):
        return False
    return True


def tile_generator(img: np.ndarray, tile: int, stride: int) -> Iterable[Tuple[np.ndarray, int, int]]:
    """Yield tiles (H, W) of size (tile, tile) with stride, dropping partial edges."""
    H, W = img.shape[-2], img.shape[-1]
    # ensure exact tiles only (no padding)
    max_y = H - tile
    max_x = W - tile
    if max_y < 0 or max_x < 0:
        return
    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            patch = img[y:y + tile, x:x + tile]
            if patch.shape == (tile, tile):
                yield patch, y, x


def save_cutout_fits(out_path: Path, arr: np.ndarray, retries: int = 5):
    """Robust FITS write with parent mkdir and retry-on-ENOENT (common on busy FS)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            hdu = fits.PrimaryHDU(data=arr)
            fits.HDUList([hdu]).writeto(out_path, overwrite=True)
            return
        except OSError as e:
            if e.errno == errno.ENOENT and attempt < retries - 1:
                # backoff to dodge dir cache/coherency glitches
                time.sleep(0.01 * (2 ** attempt) + random.random() * 0.01)
                continue
            raise


def save_cutout_npy(out_path: Path, arr: np.ndarray, retries: int = 5):
    """Robust NPY write with parent mkdir and retry-on-ENOENT."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            np.save(out_path, arr)
            return
        except OSError as e:
            if e.errno == errno.ENOENT and attempt < retries - 1:
                time.sleep(0.01 * (2 ** attempt) + random.random() * 0.01)
                continue
            raise


def process_fits_file(
    fpath: Path,
    out_tmp_dir: Path,
    tile_size: int,
    stride: int,
    out_format: str,
    do_not_scale: bool,
    plane_index: Optional[int],
) -> int:
    """
    Process a single FITS file for 2D planes. If `plane_index` is an int, processes a single 2D plane
    selected by `plane_index` over leading dims. If `plane_index` is None, processes all planes.
    Writes cutouts into `out_tmp_dir` (no batching here). Returns the number of cutouts written.
    """
    # Write into a per-worker subdirectory to avoid hot-directory contention
    try:
        worker_dir = out_tmp_dir / f"w{os.getpid():05d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        worker_dir = out_tmp_dir

    wrote = 0
    try:
        # Open with or without FITS scaling (BSCALE/BZERO)
        with fits.open(
            fpath,
            memmap=True,
            do_not_scale_image_data=do_not_scale,
            uint=True,
        ) as hdul:
            # Find first HDU with 2D-or-more image data
            chosen = None
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and hdu.data.ndim >= 2:
                    chosen = hdu
                    break
            if chosen is None:
                return 0

            data = chosen.data  # numpy memmap or ndarray

            if plane_index is None:
                # Process all planes
                if data.ndim == 2:
                    planes = [(data, 0)]
                else:
                    lead_shape = data.shape[:-2]
                    total = int(np.prod(lead_shape))
                    planes = [select_plane(data, idx) for idx in range(total)]
            else:
                planes = [select_plane(data, plane_index)]

            for plane, flat_idx in planes:
                for patch, y, x in tile_generator(plane, tile_size, stride):
                    if not valid_tile(patch):
                        continue

                    stem = fpath.stem
                    plane_tag = f"p{flat_idx:03d}"
                    # Many LoTSS tiles share the same stem (e.g., 'mosaic-blanked').
                    # Disambiguate by prefixing the immediate parent directory and suffixing a stable 8-char hash.
                    parent = fpath.parent.name
                    h8 = hashlib.md5(str(fpath).encode('utf-8')).hexdigest()[:8]
                    fname_base = f"{parent}__{stem}_{plane_tag}_y{y:05d}_x{x:05d}_{h8}"

                    if out_format == "fits":
                        out_path = worker_dir / f"{fname_base}.fits"
                        save_cutout_fits(out_path, patch)
                    else:
                        out_path = worker_dir / f"{fname_base}.npy"
                        save_cutout_npy(out_path, patch)

                    wrote += 1

    except Exception as e:
        print(f"[WARN] Skipping {fpath} due to error: {e}")

    return wrote


def consolidate_and_bucket(out_tmp_dir: Path, out_root: Path, max_per_dir: int) -> int:
    """Move files from staging to batched subfolders of size <= max_per_dir. Returns total moved."""
    files = sorted(out_tmp_dir.rglob("*"))
    files = [p for p in files if p.is_file()]
    total = 0
    bucket = 0
    in_bucket = 0
    batch_dir = None
    for f in files:
        if batch_dir is None or in_bucket >= max_per_dir:
            batch_dir = out_root / f"{bucket:06d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            bucket += 1
            in_bucket = 0
        dest = batch_dir / f.name
        shutil.move(str(f), str(dest))
        in_bucket += 1
        total += 1
    return total


def main():
    # ==== Configuration (edit here) ====
    INPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/MGCLS_images")
    OUTPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/MeerKAT512")

    # INPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/ASKAP_images")
    # OUTPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/ASKAP512")

    # INPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/LoTSS_images")
    # OUTPUT_DIR = Path("/mnt/large_volume/adema02/Datasets/LoTSS512")

    TILE_SIZE = 512
    STRIDE = 512  # set < TILE_SIZE for overlap, e.g. 50% = 256 for tiles of 512
    MAX_PER_DIR = 10000
    OUT_FORMAT = "fits"  # or "npy"
    NO_SCALE = False      # True -> disable BSCALE/BZERO
    PLANE_INDEX = None       # set an int to pick a single plane; set to None to process all planes
    NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
    # ==================================

    assert TILE_SIZE > 0 and STRIDE > 0, "tile and stride must be positive"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = OUTPUT_DIR / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    files = list(list_fits_files(INPUT_DIR))
    if not files:
        print(f"No FITS files found under {INPUT_DIR}")
        return

    print(f"Discovered {len(files)} FITS files. Using {NUM_WORKERS} workers.")

    total_written = 0
    # Multiprocessing over files; each worker writes into the shared staging dir
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(
                process_fits_file,
                fpath,
                staging_dir,
                TILE_SIZE,
                STRIDE,
                OUT_FORMAT,
                NO_SCALE,
                PLANE_INDEX,
            ): fpath
            for fpath in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing FITS files"):
            fpath = futures[fut]
            try:
                wrote = fut.result()
            except Exception as e:
                print(f"[WARN] Worker failed on {fpath}: {e}")
                wrote = 0
            total_written += wrote

    # Consolidate into batch_* subfolders
    moved = consolidate_and_bucket(staging_dir, OUTPUT_DIR, MAX_PER_DIR)
    # Cleanup staging dir (now empty); ignore errors if not empty
    try:
        staging_dir.rmdir()
    except Exception:
        pass

    print(f"Done. Total cutouts written: {total_written}. Files moved into batches: {moved}.")


if __name__ == "__main__":
    main()