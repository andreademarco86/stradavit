#!/usr/bin/env python3

"""
Radio Galaxy Zoo DR1 Cutout Downloader (FIRST Only)

This script downloads the RGZ DR1 catalog from Zenodo and fetches
corresponding FIRST survey radio cutouts using astroquery.

Cutouts are sized adaptively based on catalog angular size (LAE) with configurable padding.
Consensus-level filtering (CL) for high-confidence labels.
Enforces minimum pixel size for all cutouts.

Requirements:
    pip install requests pandas astropy astroquery tqdm

Author: Modified for RGZ DR1 with correct column names + min pixel size
Date: December 2025
"""

import os
import sys
import zipfile
import tarfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astroquery.image_cutouts.first import First
First.clear_cache()
import time
import threading
# import warnings
# from astropy.utils.exceptions import FITSFixedWarning
# warnings.filterwarnings("ignore", category=FITSFixedWarning)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

ZENODO_RECORD_ID = "14195049"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

OUTPUT_DIR = "rgz_dr1_data"
CATALOG_DIR = os.path.join(OUTPUT_DIR, "catalogs")

# Consensus filtering (CL column in RGZ DR1)
CONSENSUS_THRESHOLD = 0.65  # Minimum consensus level (0.65 is standard, None to disable)

# Morphology class filtering
FILTER_TOP_6_CLASSES = True  # Keep only the 6 main morphology classes
VALID_MORPHOLOGY_CLASSES = {
    '1c1p',  # 1 component, 1 peak
    '1c2p',  # 1 component, 2 peaks
    '1c3p',  # 1 component, 3 peaks
    '2c2p',  # 2 components, 2 peaks
    '2c3p',  # 2 components, 3 peaks
    '3c3p',  # 3 components, 3 peaks
}

# Adaptive sizing based on catalog angular size (LAE column = Largest Angular Extent)
PADDING_FACTOR = 1.5  # Multiply LAE by this (1.5 is standard from papers)

# Minimum pixel size constraint
MIN_PIXEL_SIZE = 8  # Minimum side length in pixels (ensures no ultra-tiny images)
FIRST_PIXEL_SCALE = 1.8  # FIRST pixel scale in arcsec/pixel
CENTER_TOL_ARCMIN = 1.0  # Require cutout center to match target within this tolerance

# Download limits
MAX_SOURCES = None  # Set to an integer to limit downloads (e.g., 100 for testing, None for all)
DELAY_BETWEEN_REQUESTS = 1.0  # seconds between cutout requests to avoid overloading server
MAX_WORKERS = 3  # Set to 1 to disable threading; keep small to avoid hammering the service

# Organization
ORGANIZE_BY_CLASSIFICATION = True  # Create subfolders for each classification label

# Resume support
SKIP_EXISTING_FITS = True  # Skip downloads/processing when a cutout already exists

# Robust retry configuration
MAX_RETRIES = 3  # total attempts for retryable server/network errors
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}
BACKOFF_BASE = 2.0  # exponential backoff base
BACKOFF_JITTER = 0.25  # +/-25% jitter to reduce thundering herd
BACKOFF_CAP = 10.0  # max sleep seconds between attempts
READ_TIMEOUT = 60  # seconds; used if astroquery honors TIMEOUT

# Try to raise astroquery timeout ceiling if available
try:
    First.TIMEOUT = max(getattr(First, "TIMEOUT", 30), READ_TIMEOUT)
except Exception:
    pass

# ============================================================================
# FITS VALIDATION & RETRY HELPERS
# ============================================================================

from astropy.io.fits.verify import VerifyError as _FitsVerifyError


class NoFirstDataError(RuntimeError):
    """Raised when the FIRST service reports no coverage at the requested position."""

def _is_valid_fits_hdu(hdu):
    """Return True if HDU looks like a valid 2D FIRST FITS image.
    Checks header sanity and astropy verify().
    """
    try:
        # Run FITS compliance checks
        hdu.verify("exception")
    except Exception:
        # If verify() raises, treat as invalid
        return False

    hdr = hdu.header if hasattr(hdu, "header") else None
    if hdr is None:
        return False

    # Minimal required keys
    if hdr.get("SIMPLE") not in (True, "T", "TRUE"):
        return False
    if hdr.get("NAXIS") != 2:
        return False
    if hdr.get("NAXIS1") is None or hdr.get("NAXIS2") is None:
        return False

    # Sanity: small cutouts should not be zero-sized
    naxis1 = int(hdr["NAXIS1"]) if isinstance(hdr.get("NAXIS1"), (int, float, str)) else -1
    naxis2 = int(hdr["NAXIS2"]) if isinstance(hdr.get("NAXIS2"), (int, float, str)) else -1
    if naxis1 <= 0 or naxis2 <= 0:
        return False

    # If data exists, ensure it is 2D numeric
    data = getattr(hdu, "data", None)
    if data is None or not hasattr(data, "ndim") or data.ndim != 2:
        return False

    # Treat images with no variation (all zeros / constant / NaN) as invalid
    try:
        arr = np.asarray(data)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return False
        finite_vals = arr[finite_mask]
        # If there is effectively no variation, consider it a "blank" cutout
        if np.nanstd(finite_vals) == 0:
            return False
    except Exception:
        # If stats computation fails, be conservative and accept; header checks already passed
        pass

    return True

# Robust retry helpers
import random
from urllib.error import HTTPError as _UrllibHTTPError, URLError as _UrllibURLError
import requests as _requests

def _is_retryable_error(exc):
    """Return True if exception indicates a transient server/network issue."""
    # Network layer
    if isinstance(exc, (_requests.exceptions.Timeout, _requests.exceptions.ConnectionError, _UrllibURLError)):
        return True

    # HTTP status, if present
    status = None
    # requests-style
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    # urllib-style
    if status is None and isinstance(exc, _UrllibHTTPError):
        status = getattr(exc, "code", None)

    if isinstance(status, int) and status in RETRY_HTTP_STATUSES:
        return True

    # Some astroquery wrappers include status code in the message
    msg = str(exc).lower()
    for code in ("429", "500", "502", "503", "504", "timeout", "timed out", "temporarily unavailable"):
        if code in msg:
            return True

    return False


def _is_no_first_data_error(exc):
    """Return True if astroquery reports the target is outside the FIRST footprint."""
    msg = str(exc).lower()
    return "no first data is available" in msg

def _retry_sleep(attempt):
    """Sleep with exponential backoff and jitter based on attempt index (0-based)."""
    base = DELAY_BETWEEN_REQUESTS * (BACKOFF_BASE ** attempt)
    jitter = 1.0 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
    time.sleep(min(base * jitter, BACKOFF_CAP))


def _format_exception_reason(exc):
    """Return a concise reason string for logging/CSV, capturing status codes if present."""
    status = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    if status is None and isinstance(exc, _UrllibHTTPError):
        status = getattr(exc, "code", None)

    typename = exc.__class__.__name__
    msg = str(exc).strip()
    parts = [typename]
    if status is not None:
        parts.append(f"status={status}")
    if msg:
        parts.append(msg)
    return ": ".join(parts)[:180]

def _fetch_first_with_retry(coord, cutout_size):
    """Call First.get_images with robust retry for transient errors and hard timeout."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            # Guard against hanging network calls by running with a hard timeout
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(First.get_images, coord, image_size=cutout_size)
                return fut.result(timeout=READ_TIMEOUT)
        except FuturesTimeout as e:
            last_err = e
            _retry_sleep(attempt)
            continue
        except Exception as e:
            last_err = e
            if _is_no_first_data_error(e):
                raise NoFirstDataError(_format_exception_reason(e)) from e
            if _is_retryable_error(e):
                _retry_sleep(attempt)
                continue
            # Non-retryable
            break
    # Exhausted or non-retryable
    if last_err is not None:
        if _is_no_first_data_error(last_err):
            raise NoFirstDataError(_format_exception_reason(last_err)) from last_err
        raise RuntimeError(f"FIRST request failed after {MAX_RETRIES} attempt(s): {_format_exception_reason(last_err)}") from last_err
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def arcmin_to_pixels(size_arcmin, pixel_scale_arcsec=FIRST_PIXEL_SCALE):
    """Convert angular size in arcmin to approximate pixel size."""
    size_arcsec = size_arcmin * 60.0
    return size_arcsec / pixel_scale_arcsec

def pixels_to_arcmin(size_pixels, pixel_scale_arcsec=FIRST_PIXEL_SCALE):
    """Convert pixel size to angular size in arcmin."""
    size_arcsec = size_pixels * pixel_scale_arcsec
    return size_arcsec / 60.0

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def get_cutout_folder_name():
    """Generate folder name based on padding factor and min pixel size."""
    return f"cutouts_pad{PADDING_FACTOR:.2f}_minpix{MIN_PIXEL_SIZE}"

def create_directories():
    """Create necessary directories for data storage."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CATALOG_DIR, exist_ok=True)

    # Create cutouts directory
    cutouts_dir = os.path.join(OUTPUT_DIR, get_cutout_folder_name())
    os.makedirs(cutouts_dir, exist_ok=True)

    print(f"✓ Created directories in: {OUTPUT_DIR}")
    print(f"✓ Cutouts folder: {cutouts_dir}")
    return cutouts_dir

# ============================================================================
# ZENODO CATALOG DOWNLOAD
# ============================================================================

def download_zenodo_catalog():
    """Download the RGZ DR1 catalog from Zenodo."""
    print("\n" + "=" * 70)
    print("Downloading RGZ DR1 Catalog from Zenodo...")
    print("=" * 70)

    # Get record metadata
    print(f"Fetching metadata from Zenodo record {ZENODO_RECORD_ID}...")
    response = requests.get(ZENODO_API_URL)
    response.raise_for_status()
    record_data = response.json()

    # Get file information
    files = record_data.get('files', [])
    if not files:
        raise ValueError("No files found in Zenodo record")

    print(f"Found {len(files)} file(s) in the record:")
    for file_info in files:
        filename = file_info['key']
        size_mb = file_info['size'] / (1024 * 1024)
        print(f"  - {filename} ({size_mb:.2f} MB)")

    # Download all files
    catalog_files = []
    for file_info in files:
        filename = file_info['key']
        download_url = file_info['links']['self']
        output_path = os.path.join(CATALOG_DIR, filename)

        print(f"\nDownloading {filename}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print(f"✓ Saved to: {output_path}")

        # Extract archive files
        if filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                member_list = zip_ref.namelist()
                zip_ref.extractall(CATALOG_DIR)
                print(f"✓ Extracted to: {CATALOG_DIR}")
                # List extracted files
                for extracted_file in member_list:
                    extracted_path = os.path.join(CATALOG_DIR, extracted_file)
                    if os.path.isfile(extracted_path):
                        catalog_files.append(extracted_path)

        elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            print(f"Extracting {filename}...")
            with tarfile.open(output_path, 'r:gz') as tar_ref:
                member_list = tar_ref.getmembers()
                tar_ref.extractall(CATALOG_DIR)
                print(f"✓ Extracted to: {CATALOG_DIR}")
                # List extracted files
                for member in member_list:
                    if member.isfile():
                        extracted_path = os.path.join(CATALOG_DIR, member.name)
                        catalog_files.append(extracted_path)

        elif filename.endswith('.tar'):
            print(f"Extracting {filename}...")
            with tarfile.open(output_path, 'r:') as tar_ref:
                member_list = tar_ref.getmembers()
                tar_ref.extractall(CATALOG_DIR)
                print(f"✓ Extracted to: {CATALOG_DIR}")
                # List extracted files
                for member in member_list:
                    if member.isfile():
                        extracted_path = os.path.join(CATALOG_DIR, member.name)
                        catalog_files.append(extracted_path)
        else:
            catalog_files.append(output_path)

    return catalog_files

# ============================================================================
# CATALOG LOADING & FILTERING
# ============================================================================

def load_rgz_catalog(catalog_files):
    """
    Load RGZ FIRST catalog from downloaded files.
    The RGZ DR1 contains tables for FIRST and ATLAS surveys.
    We load only the FIRST radio classifications file.
    """
    print("\n" + "=" * 70)
    print("Loading RGZ DR1 FIRST Catalog...")
    print("=" * 70)

    print(f"\nFound {len(catalog_files)} extracted file(s):")
    for f in catalog_files:
        print(f"  - {f}")

    # Look for FIRST RADIO CLASSIFICATIONS file
    first_catalog = None
    for file_path in catalog_files:
        filename = os.path.basename(file_path).lower()
        if 'first' in filename and 'radio_classification' in filename:
            print(f"\n->  Found FIRST radio classifications file: {file_path}")
            try:
                if file_path.endswith('.csv'):
                    first_catalog = pd.read_csv(file_path)
                elif file_path.endswith('.fits'):
                    with fits.open(file_path) as hdul:
                        first_catalog = pd.DataFrame(hdul[1].data)

                if first_catalog is not None:
                    print(f"-> Loaded {len(first_catalog)} sources from FIRST catalog")
                    print(f"\nCatalog columns: {list(first_catalog.columns)}")
                    print(f"\nFirst few rows:")
                    print(first_catalog.head())
                    break

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    if first_catalog is None:
        print("\n" + "=" * 70)
        print("ERROR: Could not load FIRST radio classifications catalog")
        print("=" * 70)
        print("\nExtracted files:")
        for f in catalog_files:
            print(f"  - {f}")
        sys.exit(1)

    # ========================================================================
    # CONSENSUS FILTERING (CL column = Consensus Level)
    # ========================================================================
    if CONSENSUS_THRESHOLD is not None:
        print(f"\n{'=' * 70}")
        print(f"Filtering by consensus threshold >= {CONSENSUS_THRESHOLD}")
        print(f"{'=' * 70}")

        # RGZ DR1 uses 'CL' for consensus level
        consensus_col = None
        if 'CL' in first_catalog.columns:
            consensus_col = 'CL'
        else:
            # Fallback to other possible names
            for col in ['consensus_level', 'consensus', 'confidence', 'p_match', 'agreement']:
                if col in first_catalog.columns:
                    consensus_col = col
                    break

        if consensus_col:
            initial_count = len(first_catalog)

            # Show CL statistics before filtering
            cl_stats = first_catalog[consensus_col].describe()
            print(f"\nConsensus Level (CL) statistics BEFORE filtering:")
            print(f"  Mean:   {cl_stats['mean']:.3f}")
            print(f"  Median: {first_catalog[consensus_col].median():.3f}")
            print(f"  Min:    {cl_stats['min']:.3f}")
            print(f"  Max:    {cl_stats['max']:.3f}")

            # Apply filter
            first_catalog = first_catalog[first_catalog[consensus_col] >= CONSENSUS_THRESHOLD].copy()
            filtered_count = len(first_catalog)

            print(f"\n->  Using consensus column: '{consensus_col}'")
            print(f"->  Filtered from {initial_count} to {filtered_count} sources")
            print(f"  ({100.0 * filtered_count / initial_count:.1f}% retained)")

            # Show CL statistics after filtering
            cl_stats_after = first_catalog[consensus_col].describe()
            print(f"\nConsensus Level (CL) statistics AFTER filtering:")
            print(f"  Mean:   {cl_stats_after['mean']:.3f}")
            print(f"  Median: {first_catalog[consensus_col].median():.3f}")
            print(f"  Min:    {cl_stats_after['min']:.3f}")
            print(f"  Max:    {cl_stats_after['max']:.3f}")
        else:
            print(f"->  WARNING: No consensus column found in {list(first_catalog.columns)}")
            print(f"->   Expected 'CL' column but it was not present")
            print(f"->   Proceeding with all sources (may include noisy labels)")
    else:
        print("\n• Consensus filtering disabled (CONSENSUS_THRESHOLD = None)")

    # ========================================================================
    # MORPHOLOGY FILTERING (top 6 classes: 1c1p, 1c2p, 1c3p, 2c2p, 2c3p, 3c3p)
    # ========================================================================
    if FILTER_TOP_6_CLASSES:
        print(f"\n{'=' * 70}")
        print("Filtering by morphology (top 6 classes)")
        print(f"{'=' * 70}")

        initial_count = len(first_catalog)

        # Compute morphology labels for each row
        first_catalog = first_catalog.copy()
        first_catalog['classification_label'] = first_catalog.apply(get_classification_label, axis=1)

        # Show distribution before filtering
        class_counts_before = first_catalog['classification_label'].value_counts().sort_index()
        print("\nMorphology distribution BEFORE filtering:")
        for cls_label, cnt in class_counts_before.items():
            print(f"  {cls_label:10s}: {cnt:6d}")

        # Apply filter to keep only desired morphology classes
        first_catalog = first_catalog[first_catalog['classification_label'].isin(VALID_MORPHOLOGY_CLASSES)].copy()
        filtered_count = len(first_catalog)

        print(f"\n->  Keeping only classes: {sorted(VALID_MORPHOLOGY_CLASSES)}")
        print(f"->  Filtered from {initial_count} to {filtered_count} sources")
        if initial_count > 0:
            print(f"   ({100.0 * filtered_count / initial_count:.1f}% retained)")

        # Show distribution after filtering
        class_counts_after = first_catalog['classification_label'].value_counts().sort_index()
        print("\nMorphology distribution AFTER filtering:")
        for cls_label, cnt in class_counts_after.items():
            print(f"  {cls_label:10s}: {cnt:6d}")

    return first_catalog

# ============================================================================
# CLASSIFICATION LABEL EXTRACTION
# ============================================================================

def get_classification_label(row):
    """
    Extract classification label from catalog row.
    RGZ DR1 radio classifications use format like "1c2p" (1 component, 2 peaks).
    Classification is constructed from N_comp and N_peaks columns.

    Returns: string label like "1c2p" or "unknown" if not found
    """
    # RGZ DR1 uses N_comp and N_peaks columns
    n_comp = None
    n_peak = None

    if 'N_comp' in row.index and pd.notna(row['N_comp']):
        try:
            n_comp = int(row['N_comp'])
        except:
            pass

    if 'N_peaks' in row.index and pd.notna(row['N_peaks']):
        try:
            n_peak = int(row['N_peaks'])
        except:
            pass

    if n_comp is not None and n_peak is not None:
        return f"{n_comp}c{n_peak}p"

    return "unknown"

def get_angular_size(row):
    """
    Extract angular size from catalog row.
    RGZ DR1 uses 'LAE' (Largest Angular Extent) in arcseconds.

    Returns angular size in arcseconds, or None if not found.
    """
    # RGZ DR1 column: LAE = Largest Angular Extent (arcsec)
    if 'LAE' in row.index and pd.notna(row['LAE']):
        try:
            lae = float(row['LAE'])
            if lae > 0:  # Sanity check
                return lae
        except:
            pass

    # Fallback to other possible column names
    size_cols = [
        'angular_size', 'Angular_size', 'size', 'extent',
        'major_axis', 'LAS', 'D'
    ]

    for col in size_cols:
        if col in row.index and pd.notna(row[col]):
            try:
                size_val = float(row[col])
                if size_val > 0:
                    return size_val
            except:
                continue

    return None

# ============================================================================
# COORDINATE PARSING HELPERS
# ============================================================================

def parse_rgzid_coordinates(rgzid_str):
    """
    Parse an RGZID like 'J013744.1+044004' into (ra_deg, dec_deg).
    Returns None if parsing fails.
    """
    if rgzid_str is None:
        return None
    s = str(rgzid_str).strip()
    if not s or 'J' not in s:
        return None
    try:
        body = s[s.index('J') + 1:]
        if '+' in body:
            ra_str, dec_str = body.split('+', 1)
            dec_sign = 1.0
        elif '-' in body:
            ra_str, dec_str = body.split('-', 1)
            dec_sign = -1.0
        else:
            return None

        if len(ra_str) < 6:
            return None
        ra_h = int(ra_str[0:2])
        ra_m = int(ra_str[2:4])
        ra_s = float(ra_str[4:])
        ra_hours = ra_h + ra_m / 60.0 + ra_s / 3600.0
        ra_deg = ra_hours * 15.0

        if len(dec_str) < 4:
            return None
        dec_d = int(dec_str[0:2])
        dec_m = int(dec_str[2:4]) if len(dec_str) >= 4 else 0
        dec_s = float(dec_str[4:]) if len(dec_str) > 4 else 0.0
        dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)

        return ra_deg, dec_deg
    except Exception:
        return None

# Consider catalog RA/Dec inconsistent with RGZID if separation exceeds this (arcmin)
COORD_MISMATCH_ARCMIN = 1.0


def _cutout_center_separation_arcmin(hdu, target_coord: SkyCoord):
    """Compute separation (arcmin) between requested coord and cutout WCS center."""
    try:
        w = WCS(hdu.header)
        data = getattr(hdu, "data", None)
        if data is None or not hasattr(data, "shape") or len(data.shape) != 2:
            return None
        ny, nx = data.shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
        world = w.pixel_to_world(cx, cy)
        return target_coord.separation(world).arcmin
    except Exception:
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_classification_subfolder(classification_label, cutouts_base_dir):
    """Create subfolder for a classification label inside the cutouts directory."""
    # Sanitize label for folder name
    safe_label = str(classification_label).replace('/', '_').replace('\\', '_')
    safe_label = safe_label.replace(' ', '_').replace(':', '_')

    subfolder_path = os.path.join(cutouts_base_dir, safe_label)
    os.makedirs(subfolder_path, exist_ok=True)
    return subfolder_path


def _cutout_cache_key(path_like):
    """Normalize a folder path into a cache key."""
    return str(Path(path_like).resolve())


def build_existing_cutout_index(cutouts_base_dir):
    """Index existing FITS files by folder for resume support."""
    base_path = Path(cutouts_base_dir)
    search_dirs = [base_path]
    if ORGANIZE_BY_CLASSIFICATION:
        search_dirs = [p for p in base_path.iterdir() if p.is_dir()]

    cache = {}
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        names = set(f.name for f in directory.glob("*.fits"))
        prefixes = set()
        for name in names:
            # Capture the id prefix before the _ra... portion for robustness
            prefixes.add(name.split("_ra", 1)[0])
        cache[_cutout_cache_key(directory)] = {"names": names, "prefixes": prefixes}
    return cache


def should_skip_existing(cache, output_dir, output_filename, safe_id):
    """Return True if output FITS already exists (by full name or id prefix)."""
    entry = cache.setdefault(_cutout_cache_key(output_dir), {"names": set(), "prefixes": set()})
    if output_filename in entry["names"]:
        return True
    if safe_id in entry["prefixes"]:
        return True
    return any(name.startswith(f"{safe_id}_") for name in entry["names"])


def record_cutout(cache, output_dir, output_filename, safe_id):
    """Record a newly created cutout in the cache to avoid duplicate work."""
    entry = cache.setdefault(_cutout_cache_key(output_dir), {"names": set(), "prefixes": set()})
    entry["names"].add(output_filename)
    entry["prefixes"].add(safe_id)

# ============================================================================
# CUTOUT DOWNLOAD (ADAPTIVE SIZING WITH MIN PIXEL SIZE)
# ============================================================================

def _process_single_source(idx, row, ra_col, dec_col, min_cutout_for_pixels, cutouts_base_dir,
                           existing_cutouts_cache, cache_lock):
    """Worker to download a single cutout; returns a dict with stats/metadata."""
    classification = get_classification_label(row)
    result = {
        "classification": None,
        "metadata": None,
        "failed": None,
        "size_stat": None,
        "pixel_stat": None,
        "missing_lae": False,
        "skipped_invalid_lae": False,
        "upscaled_for_minpix": False,
        "coord_mismatch": False,
        "center_mismatch": False,
        "resume_skip": False,
    }

    try:
        ra = dec = None
        if ra_col and dec_col:
            try:
                ra = float(row[ra_col])
                dec = float(row[dec_col])
            except (ValueError, TypeError):
                pass

        rgz_coord = None
        if 'RGZID' in row.index and pd.notna(row['RGZID']):
            rgz_coord = parse_rgzid_coordinates(row['RGZID'])

        coord_source = "catalog"
        coord_mismatch_flag = False
        if rgz_coord is not None:
            if ra is None or dec is None:
                ra, dec = rgz_coord
                coord_source = "rgzid"
            else:
                try:
                    sep = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs').separation(
                        SkyCoord(ra=rgz_coord[0] * u.deg, dec=rgz_coord[1] * u.deg, frame='icrs')
                    )
                    if sep.arcmin > COORD_MISMATCH_ARCMIN:
                        ra, dec = rgz_coord
                        coord_source = "rgzid"
                        coord_mismatch_flag = True
                    else:
                        coord_mismatch_flag = False
                except Exception:
                    pass

        if ra is None or dec is None:
            return result  # Skip rows with invalid coordinates (matches original behavior)

        result["classification"] = classification
        result["coord_mismatch"] = coord_mismatch_flag

        angular_size_arcsec = get_angular_size(row)
        if angular_size_arcsec is None or angular_size_arcsec <= 0:
            result["missing_lae"] = True
            result["skipped_invalid_lae"] = True
            result["metadata"] = {
                'source_idx': idx,
                'source_id': None,
                'ra': ra,
                'dec': dec,
                'classification': classification,
                'LAE_arcsec': angular_size_arcsec,
                'cutout_size_arcmin': None,
                'estimated_pixels': None,
                'filename': None,
                'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                'status': 'skipped_invalid_LAE',
                'coord_source': coord_source,
                'center_sep_arcmin': None
            }
            return result

        cutout_size_arcmin = (angular_size_arcsec / 60.0) * PADDING_FACTOR
        estimated_pixels = arcmin_to_pixels(cutout_size_arcmin)
        upscaled = False
        if estimated_pixels < MIN_PIXEL_SIZE:
            cutout_size_arcmin = max(cutout_size_arcmin, min_cutout_for_pixels)
            estimated_pixels = arcmin_to_pixels(cutout_size_arcmin)
            upscaled = True

        cutout_size = cutout_size_arcmin * u.arcmin
        result["size_stat"] = cutout_size_arcmin
        result["pixel_stat"] = estimated_pixels
        result["upscaled_for_minpix"] = upscaled

        if ORGANIZE_BY_CLASSIFICATION:
            output_dir = create_classification_subfolder(classification, cutouts_base_dir)
        else:
            output_dir = cutouts_base_dir

        source_id = f"RGZ_{idx:06d}"
        if 'RGZID' in row.index:
            try:
                source_id = row['RGZID']
            except Exception:
                pass
        elif 'ZooniverseID' in row.index:
            try:
                source_id = row['ZooniverseID']
            except Exception:
                pass

        safe_id = str(source_id).replace('/', '_').replace(' ', '_')
        output_filename = f"{safe_id}_ra{ra:.4f}_dec{dec:.4f}.fits"
        output_path = os.path.join(output_dir, output_filename)

        existing_hit = False
        with cache_lock:
            if SKIP_EXISTING_FITS and should_skip_existing(existing_cutouts_cache, output_dir, output_filename, safe_id):
                existing_hit = True
            elif os.path.exists(output_path):
                existing_hit = True

            if existing_hit:
                record_cutout(existing_cutouts_cache, output_dir, output_filename, safe_id)

        if existing_hit:
            result["resume_skip"] = True
            result["metadata"] = {
                'source_idx': idx,
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'classification': classification,
                'LAE_arcsec': angular_size_arcsec,
                'cutout_size_arcmin': cutout_size_arcmin,
                'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                'filename': output_filename,
                'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                'status': 'already_exists',
                'coord_source': coord_source,
                'center_sep_arcmin': None
            }
            return result

        issued_request = False
        center_sep_arcmin = None
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

            issued_request = True
            image_list = _fetch_first_with_retry(coord, cutout_size)

            if image_list and len(image_list) > 0:
                candidate = image_list[0]
                if isinstance(candidate, fits.HDUList):
                    hdu = candidate[0]
                else:
                    hdu = candidate
            else:
                hdu = None

            if hdu is None or not _is_valid_fits_hdu(hdu):
                result["failed"] = {
                    'source_idx': idx,
                    'source_id': source_id,
                    'ra': ra,
                    'dec': dec,
                    'classification': classification,
                    'LAE_arcsec': angular_size_arcsec,
                    'cutout_size_arcmin': cutout_size_arcmin,
                    'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                    'reason': 'invalid_fits_header_or_payload',
                    'coord_source': coord_source,
                    'center_sep_arcmin': None
                }
                result["metadata"] = {
                    'source_idx': idx,
                    'source_id': source_id,
                    'ra': ra,
                    'dec': dec,
                    'classification': classification,
                    'LAE_arcsec': angular_size_arcsec,
                    'cutout_size_arcmin': cutout_size_arcmin,
                    'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                    'filename': None,
                    'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                    'status': 'invalid_fits',
                    'coord_source': coord_source,
                    'center_sep_arcmin': None
                }
                return result

            center_sep_arcmin = _cutout_center_separation_arcmin(hdu, coord)
            if center_sep_arcmin is not None and center_sep_arcmin > CENTER_TOL_ARCMIN:
                result["center_mismatch"] = True
                result["failed"] = {
                    'source_idx': idx,
                    'source_id': source_id,
                    'ra': ra,
                    'dec': dec,
                    'classification': classification,
                    'LAE_arcsec': angular_size_arcsec,
                    'cutout_size_arcmin': cutout_size_arcmin,
                    'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                    'reason': f'center_mismatch_{center_sep_arcmin:.2f}arcmin',
                    'coord_source': coord_source,
                }
                result["metadata"] = {
                    'source_idx': idx,
                    'source_id': source_id,
                    'ra': ra,
                    'dec': dec,
                    'classification': classification,
                    'LAE_arcsec': angular_size_arcsec,
                    'cutout_size_arcmin': cutout_size_arcmin,
                    'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                    'filename': None,
                    'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                    'status': 'center_mismatch',
                    'coord_source': coord_source,
                    'center_sep_arcmin': center_sep_arcmin,
                }
                return result

            hdu.writeto(output_path, overwrite=True)

            result["metadata"] = {
                'source_idx': idx,
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'classification': classification,
                'LAE_arcsec': angular_size_arcsec,
                'cutout_size_arcmin': cutout_size_arcmin,
                'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                'filename': output_filename,
                'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                'status': 'success',
                'coord_source': coord_source,
                'center_sep_arcmin': center_sep_arcmin
            }

            with cache_lock:
                record_cutout(existing_cutouts_cache, output_dir, output_filename, safe_id)

            return result

        except Exception as e:
            error_msg = _format_exception_reason(e)
            status = 'error'
            if isinstance(e, NoFirstDataError):
                status = 'no_first_coverage'
            result["failed"] = {
                'source_idx': idx,
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'classification': classification,
                'LAE_arcsec': angular_size_arcsec,
                'cutout_size_arcmin': cutout_size_arcmin,
                'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                'reason': error_msg,
                'coord_source': coord_source,
                'center_sep_arcmin': center_sep_arcmin
            }
            result["metadata"] = {
                'source_idx': idx,
                'source_id': source_id,
                'ra': ra,
                'dec': dec,
                'classification': classification,
                'LAE_arcsec': angular_size_arcsec,
                'cutout_size_arcmin': cutout_size_arcmin,
                'estimated_pixels': int(arcmin_to_pixels(cutout_size_arcmin)),
                'filename': None,
                'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
                'status': status,
                'coord_source': coord_source,
                'center_sep_arcmin': center_sep_arcmin,
                'reason': error_msg
            }
            return result
        finally:
            if issued_request:
                time.sleep(DELAY_BETWEEN_REQUESTS)

    except Exception as e:
        # Catch-all to prevent thread from killing the pool
        result["classification"] = classification
        result["metadata"] = {
            'source_idx': idx,
            'source_id': None,
            'ra': None,
            'dec': None,
            'classification': classification,
            'LAE_arcsec': None,
            'cutout_size_arcmin': None,
            'estimated_pixels': None,
            'filename': None,
            'subfolder': classification if ORGANIZE_BY_CLASSIFICATION else '',
            'status': 'error',
            'coord_source': 'unknown',
            'center_sep_arcmin': None,
            'reason': str(e)[:100],
        }
        return result


def download_first_cutouts(catalog, cutouts_base_dir, max_sources=None):
    """
    Download FIRST cutouts for sources in the catalog using astroquery.

    Cutout size is determined adaptively from catalog LAE (Largest Angular Extent):
        cutout_size = PADDING_FACTOR * LAE (in arcsec) / 60 (convert to arcmin)

    Then enforces minimum pixel size:
        if cutout_size_pixels < MIN_PIXEL_SIZE:
            cutout_size = pixels_to_arcmin(MIN_PIXEL_SIZE)

    Organizes cutouts into subfolders by classification label.

    Parameters
    ----------
    catalog : pandas.DataFrame
        RGZ catalog with RA/Dec, classification, and LAE columns
    cutouts_base_dir : str
        Base directory for cutouts (e.g., 'rgz_dr1_data/cutouts_pad1.50_minpix16')
    max_sources : int, optional
        Maximum number of sources to download (for testing)
    """
    print("\n" + "=" * 70)
    print("Downloading FIRST Cutouts (Adaptive Sizing)")
    print("=" * 70)

    # Identify RA/Dec columns
    ra_col = 'RA' if 'RA' in catalog.columns else None
    dec_col = 'Dec' if 'Dec' in catalog.columns else None

    if ra_col is None or dec_col is None:
        print(f"ERROR: Could not identify RA/Dec columns.")
        print(f"Available columns: {list(catalog.columns)}")
        sys.exit(1)

    # Compute minimum cutout size to satisfy MIN_PIXEL_SIZE
    min_cutout_for_pixels = pixels_to_arcmin(MIN_PIXEL_SIZE)

    print(f"Using columns: RA={ra_col}, Dec={dec_col}")
    print(f"Adaptive sizing: cutout_size = {PADDING_FACTOR} × LAE (arcsec)")
    print(f"Minimum pixel size: {MIN_PIXEL_SIZE} px (≈ {min_cutout_for_pixels:.3f} arcmin)")
    print(f"FIRST pixel scale: {FIRST_PIXEL_SCALE} arcsec/pixel")
    print(f"Coordinate handling: prefer catalog RA/Dec, fall back to RGZID if mismatch > {COORD_MISMATCH_ARCMIN} arcmin")

    if ORGANIZE_BY_CLASSIFICATION:
        print(f"Organizing by classification labels into subfolders")

    print(f"Base cutouts directory: {cutouts_base_dir}")

    # Pre-index existing FITS files for resume support
    existing_cutouts_cache = build_existing_cutout_index(cutouts_base_dir) if SKIP_EXISTING_FITS else {}
    existing_total = 0
    if SKIP_EXISTING_FITS:
        existing_total = sum(len(v["names"]) for v in existing_cutouts_cache.values())
        print(f"Resume enabled: {existing_total} existing FITS file(s) will be skipped if matched")

    # Limit sources if specified
    sources = catalog.head(max_sources) if max_sources else catalog
    total_sources = len(sources)
    print(f"\nDownloading cutouts for {total_sources} sources...")

    # Track statistics
    classification_counts = {}
    size_stats = []
    pixel_size_stats = []
    metadata = []
    failed_downloads = []
    missing_lae_count = 0
    skipped_invalid_lae_count = 0
    upscaled_for_minpix_count = 0
    coord_mismatch_count = 0
    center_mismatch_count = 0
    resume_skip_count = 0

    cache_lock = threading.Lock()

    def handle_result(result):
        nonlocal missing_lae_count, skipped_invalid_lae_count, upscaled_for_minpix_count
        nonlocal coord_mismatch_count, center_mismatch_count, resume_skip_count

        if not result:
            return

        classification = result.get("classification")
        if classification is not None:
            classification_counts[classification] = classification_counts.get(classification, 0) + 1

        meta_entry = result.get("metadata")
        if meta_entry:
            metadata.append(meta_entry)

        failed_entry = result.get("failed")
        if failed_entry:
            failed_downloads.append(failed_entry)
            # Emit a lightweight reason so failures are visible during the run
            sid = failed_entry.get("source_id") or failed_entry.get("source_idx")
            reason = failed_entry.get("reason", "unknown_error")
            print(f"[fail] source={sid} reason={reason}")

        if result.get("size_stat") is not None:
            size_stats.append(result["size_stat"])
        if result.get("pixel_stat") is not None:
            pixel_size_stats.append(result["pixel_stat"])

        missing_lae_count += int(bool(result.get("missing_lae")))
        skipped_invalid_lae_count += int(bool(result.get("skipped_invalid_lae")))
        upscaled_for_minpix_count += int(bool(result.get("upscaled_for_minpix")))
        coord_mismatch_count += int(bool(result.get("coord_mismatch")))
        center_mismatch_count += int(bool(result.get("center_mismatch")))
        resume_skip_count += int(bool(result.get("resume_skip")))

    worker_count = 1
    if MAX_WORKERS is not None:
        try:
            worker_count = max(1, int(MAX_WORKERS))
        except Exception:
            worker_count = 1
    use_thread_pool = worker_count > 1

    if use_thread_pool:
        print(f"Parallel downloads: up to {worker_count} worker threads with ~{DELAY_BETWEEN_REQUESTS}s pause after each request")
    else:
        print("Parallel downloads disabled (MAX_WORKERS <= 1)")

    if use_thread_pool:
        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, row in sources.iterrows():
                futures.append(
                    executor.submit(
                        _process_single_source,
                        idx,
                        row,
                        ra_col,
                        dec_col,
                        min_cutout_for_pixels,
                        cutouts_base_dir,
                        existing_cutouts_cache,
                        cache_lock,
                    )
                )

            with tqdm(total=total_sources, desc="Downloading cutouts") as pbar:
                for fut in as_completed(futures):
                    handle_result(fut.result())
                    pbar.update(1)
    else:
        with tqdm(total=total_sources, desc="Downloading cutouts") as pbar:
            for idx, row in sources.iterrows():
                handle_result(
                    _process_single_source(
                        idx,
                        row,
                        ra_col,
                        dec_col,
                        min_cutout_for_pixels,
                        cutouts_base_dir,
                        existing_cutouts_cache,
                        cache_lock,
                    )
                )
                pbar.update(1)

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    if not metadata_df.empty and 'source_idx' in metadata_df.columns:
        metadata_df = metadata_df.sort_values("source_idx").drop(columns=["source_idx"]).reset_index(drop=True)
    folder_suffix = get_cutout_folder_name()
    metadata_path = os.path.join(OUTPUT_DIR, f'download_metadata_{folder_suffix}.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n-> Saved metadata to: {metadata_path}")

    # Save failed downloads
    if failed_downloads:
        failed_df = pd.DataFrame(failed_downloads)
        if 'source_idx' in failed_df.columns:
            failed_df = failed_df.sort_values("source_idx").drop(columns=["source_idx"]).reset_index(drop=True)
        failed_path = os.path.join(OUTPUT_DIR, f'failed_downloads_{folder_suffix}.csv')
        failed_df.to_csv(failed_path, index=False)
        print(f"-> Saved failed downloads to: {failed_path}")

    # Save classification distribution
    if classification_counts:
        class_dist_path = os.path.join(OUTPUT_DIR, f'classification_distribution_{folder_suffix}.csv')
        class_dist_df = pd.DataFrame([
            {'classification': k, 'count': v}
            for k, v in sorted(classification_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        class_dist_df.to_csv(class_dist_path, index=False)
        print(f"✓ Saved classification distribution to: {class_dist_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)

    success_count = len([m for m in metadata if m['status'] == 'success'])
    total_on_disk = existing_total + success_count
    print(f"Successfully downloaded: {success_count}/{total_sources}")
    print(f"Already on disk before run: {existing_total}")
    print(f"Total files on disk (existing + new): {total_on_disk}")
    if resume_skip_count:
        print(f"Skipped (already existed): {resume_skip_count}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Cutouts saved to: {cutouts_base_dir}/ (organized by classification)")

    if missing_lae_count > 0:
        print(f"\n• {missing_lae_count} sources skipped due to missing or invalid LAE")

    if upscaled_for_minpix_count > 0:
        print(f"✓ {upscaled_for_minpix_count} sources upscaled to meet MIN_PIXEL_SIZE={MIN_PIXEL_SIZE} px")

    if coord_mismatch_count > 0:
        print(f"\n✓ {coord_mismatch_count} sources used RGZID-derived coordinates due to catalog mismatch (> {COORD_MISMATCH_ARCMIN} arcmin)")

    if center_mismatch_count > 0:
        print(f"\n✓ {center_mismatch_count} cutouts rejected due to center mismatch (> {CENTER_TOL_ARCMIN} arcmin)")

    if size_stats:
        print(f"\nCutout size statistics (arcmin):")
        print(f"  Mean:   {np.mean(size_stats):.3f}")
        print(f"  Median: {np.median(size_stats):.3f}")
        print(f"  Std:    {np.std(size_stats):.3f}")
        print(f"  Min:    {np.min(size_stats):.3f}")
        print(f"  Max:    {np.max(size_stats):.3f}")

    if pixel_size_stats:
        print(f"\nEstimated pixel size statistics:")
        print(f"  Mean:   {np.mean(pixel_size_stats):.1f} px")
        print(f"  Median: {np.median(pixel_size_stats):.1f} px")
        print(f"  Min:    {np.min(pixel_size_stats):.1f} px")
        print(f"  Max:    {np.max(pixel_size_stats):.1f} px")

    if classification_counts:
        print(f"\nClassification distribution (top 10):")
        sorted_classes = sorted(classification_counts.items(), key=lambda x: x[1], reverse=True)
        for class_label, count in sorted_classes[:10]:
            print(f"  {class_label:10s}: {count:6d}")
        if len(sorted_classes) > 10:
            print(f"  ... and {len(sorted_classes) - 10} more classes")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Radio Galaxy Zoo DR1 Cutout Downloader (FIRST)")
    print("Adaptive sizing based on LAE (Largest Angular Extent)")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Padding factor: {PADDING_FACTOR}×")
    print(f"Minimum pixel size: {MIN_PIXEL_SIZE} px")
    print(f"Consensus threshold (CL): {CONSENSUS_THRESHOLD if CONSENSUS_THRESHOLD else 'Disabled'}")
    print(f"Morphology filter: {'Enabled (6 classes)' if FILTER_TOP_6_CLASSES else 'Disabled'}")

    if MAX_SOURCES:
        print(f"Limiting to first {MAX_SOURCES} sources (for testing)")
    if ORGANIZE_BY_CLASSIFICATION:
        print(f"Organizing cutouts by classification labels")

    try:
        # Step 1: Create directories
        cutouts_dir = create_directories()

        # Step 2: Download catalog from Zenodo
        catalog_files = download_zenodo_catalog()

        # Step 3: Load the catalog with consensus filtering
        catalog = load_rgz_catalog(catalog_files)

        # Step 4: Download FIRST cutouts with adaptive sizing
        download_first_cutouts(catalog, cutouts_dir, max_sources=MAX_SOURCES)

        print("\n" + "=" * 70)
        print("-> All done!")
        print("=" * 70)
        print(f"\nData saved in: {os.path.abspath(OUTPUT_DIR)}/")
        print(f"  - catalogs/ : RGZ DR1 catalog tables")
        print(f"  - {get_cutout_folder_name()}/ : FIRST FITS cutouts organized by classification")
        print(f"    (subfolders: 1c1p/, 1c2p/, 2c2p/, 3c3p/, etc.)")
        print(f"  - download_metadata_*.csv : Download status for each source")
        print(f"  - classification_distribution_*.csv : Count of sources per class")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
