import os, tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import torch
from astropy.io import fits
from tqdm import tqdm
import multiprocessing as mp
from typing import Tuple
try:
    from utils.radioastroprocessor import RadioFoundationProcessor
except ModuleNotFoundError:
    from radioastroprocessor import RadioFoundationProcessor
import random
import numpy as np, os

# Directory where this script resides (for logging)
SCRIPT_DIR = Path(__file__).resolve().parent

SRC = [
    Path("/mnt/large_volume/adema02/Datasets/MeerKAT512"),
    Path("/mnt/large_volume/adema02/Datasets/ASKAP512"),
    Path("/mnt/large_volume/adema02/Datasets/LoTSS512"),
    Path("/mnt/large_volume/adema02/Datasets/SKA512"),
]

DST_RAW = [
    Path("/mnt/large_volume/adema02/Datasets/ssl/MeerKAT512_pt"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/ASKAP512_pt"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/LoTSS512_pt"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/SKA512_pt"),
]

DST_PROCESSED = [
    Path("/mnt/large_volume/adema02/Datasets/ssl/MeerKAT512_pt_processed"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/ASKAP512_pt_processed"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/LoTSS512_pt_processed"),
    Path("/mnt/large_volume/adema02/Datasets/ssl/SKA512_pt_processed"),
]

# Single output directory for mixed telescope blobs
BIN_FILE = Path("/mnt/large_volume/adema02/Datasets/ssl/strada_images_fp32_v5.bin") # astropy zscale


# === Global image layout for all writers/readers ===
C, H, W = 1, 512, 512
DTYPE = np.float32
RECORD_BYTES = C * H * W * np.dtype(DTYPE).itemsize  # bytes per record (DTYPE, 1x512x512)

# --- Logging helpers for FITS ingestion ---
def _log_bad_fits(reason: str, fp: Path, extra: str | None = None, tel_idx: int | None = None):
    """Append the FITS path (and optional extra info) to per-reason logs,
    and, if tel_idx is provided, also to a per-telescope log file.
    This is called from worker processes; simple file append is used.
    """
    try:
        # Global per-reason log
        log_path = SCRIPT_DIR / f"badfits_{reason}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            if extra is None:
                f.write(f"{fp}\n")
            else:
                f.write(f"{fp} | {extra}\n")
        # Per-telescope log (optional)
        if tel_idx is not None:
            tel_path = SCRIPT_DIR / f"badfits_{reason}_tel{tel_idx:02d}.txt"
            with open(tel_path, "a") as f:
                if extra is None:
                    f.write(f"{fp}\n")
                else:
                    f.write(f"{fp} | {extra}\n")
    except Exception:
        pass

# --- Determine telescope index from file path ---
def _detect_tel_index(fp: Path) -> int | None:
    """Return index into SRC for the directory that contains fp, or None if not found."""
    try:
        fpr = fp.resolve()
    except Exception:
        fpr = fp
    for i, root in enumerate(SRC):
        try:
            if fpr.is_relative_to(root.resolve()):
                return i
        except AttributeError:
            # Python <3.9 fallback
            try:
                fpr.relative_to(root.resolve())
                return i
            except Exception:
                pass
        except Exception:
            pass
    return None

# --- Summary helper for FITS ingestion logs ---
def summarize_bad_fits_logs(log_dir: Path = SCRIPT_DIR):
    """Print a one-line summary of skipped FITS files by reason.
    Looks for files named badfits_*.txt in `log_dir`.
    """
    reasons = ["nodata", "nonnumeric", "empty", "naninf", "exception"]
    # Global totals
    counts = {}
    total_bad = 0
    for r in reasons:
        p = log_dir / f"badfits_{r}.txt"
        c = 0
        if p.exists():
            try:
                c = sum(1 for line in p.read_text().splitlines() if line.strip())
            except Exception:
                c = 0
        counts[r] = c
        total_bad += c
    parts = [f"{r}={counts[r]}" for r in reasons]
    print(f"[FITS/INGEST] Skipped bad FITS total={total_bad} (" + ", ".join(parts) + ")")

    # Per-telescope breakdown
    tel_counts = []  # list of dicts per telescope
    for i, root in enumerate(SRC):
        tcount = {}
        tsum = 0
        for r in reasons:
            p = log_dir / f"badfits_{r}_tel{i:02d}.txt"
            c = 0
            if p.exists():
                try:
                    c = sum(1 for line in p.read_text().splitlines() if line.strip())
                except Exception:
                    c = 0
            tcount[r] = c
            tsum += c
        tel_counts.append((i, root.name, tsum, tcount))

    for i, name, tsum, tcount in tel_counts:
        parts = ", ".join(f"{r}={tcount[r]}" for r in reasons)
        print(f"[FITS/INGEST] tel{i:02d}({name}) skipped={tsum} (" + parts + ")")

# === Filtering helpers for .pt validity before writing big binary ===
def _validate_pt_for_write(pt_path: Path,
                           low_std_thresh: float = 1e-3,
                           blank_eps: float = 1e-12) -> tuple[int, float]:
    """
    Quickly load a .pt and decide classification for writing.

    Returns (status_code, std), where status_code:
      0 = OK (finite, non-blank, std >= low_std_thresh)
      1 = BLANK (std <= blank_eps or all values equal)
      2 = LOW_STD (finite, non-blank but std < low_std_thresh)
      3 = INVALID (shape/type/non-finite)
    """
    try:
        t = torch.load(str(pt_path), map_location="cpu")
        # Normalize to CHW and keep only the first channel
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] == 1:
                pass
            elif t.shape[0] == 3:
                t = t[:1]
            elif t.shape[-1] == 3:
                t = t.permute(2, 0, 1)[:1]
            else:
                return 3, 0.0
        else:
            return 3, 0.0

        if t.shape != (1, H, W):
            return 3, 0.0

        arr = t[0].to(torch.float32).cpu().numpy()
        if not np.isfinite(arr).all():
            return 3, 0.0

        sd = float(arr.std())
        # Explicit blank detection (all equal or numerically zero variance)
        if sd <= blank_eps or float(arr.min()) == float(arr.max()):
            return 1, sd
        # Low-std but not blank
        if sd < low_std_thresh:
            return 2, sd
        return 0, sd
    except Exception:
        return 3, 0.0

def _filter_ok_worker(args):
    """
    Top-level worker wrapper for multiprocessing so it is picklable.
    Returns (path, status_code).
    """
    p, low_std_thresh, blank_eps = args
    try:
        status, _ = _validate_pt_for_write(p, low_std_thresh=low_std_thresh, blank_eps=blank_eps)
        return str(p), int(status)
    except Exception:
        return str(p), 3

def _filter_tel_lists(tel_lists: list[list[Path]],
                      workers: int = 4,
                      low_std_thresh: float = 1e-3,
                      blank_eps: float = 1e-12,
                      log_dir: Path | None = None) -> tuple[list[list[Path]], list[int]]:
    """
    Validate all candidate .pt paths and return filtered lists per telescope plus per-telescope drop counts.
    Writes logs into `log_dir` (default: SCRIPT_DIR). Behavior:
      - BLANK / INVALID are **excluded** and logged to bad_telXX.txt
      - LOW_STD are **kept** but logged to lowstd_telXX.txt for later review
      - OK are kept (no extra log)

    Implementation note:
      We may use `imap_unordered(...)` for throughput, but we always map results
      back to the original input paths before assigning to per-telescope buckets,
      so telescope attribution is order-independent.
    """
    # Flatten with telescope ownership
    flat: list[Path] = []
    owner: list[int] = []
    for ti, lst in enumerate(tel_lists):
        for p in lst:
            flat.append(p)
            owner.append(ti)

    if not flat:
        return [[] for _ in tel_lists], [0 for _ in tel_lists]

    # Run classification (map by path so we don't depend on result ordering)
    results_by_path: dict[str, int] = {}
    if workers and workers > 1:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()
        with ctx.Pool(processes=int(workers)) as pool:
            for res in pool.imap_unordered(
                    _filter_ok_worker,
                    ((p, low_std_thresh, blank_eps) for p in flat),
                    chunksize=1024):
                p_str, status = res
                results_by_path[p_str] = int(status)
    else:
        for p in flat:
            status, _ = _validate_pt_for_write(p, low_std_thresh=low_std_thresh, blank_eps=blank_eps)
            results_by_path[str(p)] = int(status)

    # Prepare per-telescope buckets
    filtered: list[list[Path]] = [[] for _ in tel_lists]
    dropped_bad: list[list[Path]] = [[] for _ in tel_lists]     # BLANK or INVALID
    lowstd_log: list[list[Path]] = [[] for _ in tel_lists]      # LOW_STD (kept)

    for p, ti in zip(flat, owner):
        status = int(results_by_path.get(str(p), 3))
        if status == 0:           # OK
            filtered[ti].append(p)
        elif status == 2:         # LOW_STD -> keep but log
            filtered[ti].append(p)
            lowstd_log[ti].append(p)
        else:                     # BLANK(1) or INVALID(3)
            dropped_bad[ti].append(p)

    # Logging
    log_root = log_dir if log_dir is not None else SCRIPT_DIR
    try:
        log_root.mkdir(parents=True, exist_ok=True)
        for ti in range(len(tel_lists)):
            # bad (blank/invalid)
            bad = dropped_bad[ti]
            if bad:
                (log_root / f"bad_tel{ti:02d}.txt").write_text(
                    "\n".join(str(p) for p in bad) + "\n"
                )
            # low std (kept)
            lows = lowstd_log[ti]
            if lows:
                (log_root / f"lowstd_tel{ti:02d}.txt").write_text(
                    "\n".join(str(p) for p in lows) + "\n"
                )
    except Exception:
        pass

    drop_counts = [len(x) for x in dropped_bad]
    return filtered, drop_counts

# Ensure each raw and processed output directory exists
for d in DST_RAW:
    d.mkdir(exist_ok=True, parents=True)
for d in DST_PROCESSED:
    d.mkdir(exist_ok=True, parents=True)


def convert_one(args: Tuple[int, Path, Path]):
    """
    Convert a FITS file to a single-channel tensor and save it atomically
    using a deterministic index-based filename (<idx>.pt).

    Output format:
      - a Torch tensor of shape (1, H, W) stored as float32
      - values are the FITS pixel values (no 3-channel expansion)

    Returns True on success, None if skipped/failed.
    """
    idx, fp, dst_dir = args
    try:
        tel_idx = _detect_tel_index(fp)
        with fits.open(fp, memmap=True) as hdul:
            hdu = hdul[0]
            data = hdu.data
            if data is None and len(hdul) > 1:
                hdu = hdul[1]
                data = hdu.data  # fallback if primary HDU empty

            if data is None:
                print(f"[WARN] {fp} has no data array.")
                _log_bad_fits("nodata", fp, tel_idx=tel_idx)
                return None

            if not np.issubdtype(np.asarray(data).dtype, np.number):
                dtype_str = str(np.asarray(data).dtype)
                print(f"[WARN] {fp} contains non-numeric data: {dtype_str}")
                _log_bad_fits("nonnumeric", fp, dtype_str, tel_idx)
                return None

            # Cast to float32 for tensor compatibility while preserving numeric values.
            data = np.asarray(data, dtype=np.float32)

            if data.size == 0:
                print(f"[WARN] {fp} is empty.")
                _log_bad_fits("empty", fp, tel_idx=tel_idx)
                return None

            # Exclude FITS with NaN/Inf pixels at source
            if not np.isfinite(data).all():
                print(f"[WARN] {fp} contains NaN/Inf; excluding from RAW conversion.")
                try:
                    n_nonfinite = int(np.size(data) - np.isfinite(data).sum())
                    _log_bad_fits("naninf", fp, f"nonfinite={n_nonfinite}", tel_idx)
                except Exception:
                    _log_bad_fits("naninf", fp, tel_idx=tel_idx)
                return None

            # Save as single-channel (raw data only) tensor
            tensor = torch.from_numpy(data).unsqueeze(0)

            out_path = dst_dir / f"{idx:08d}.pt"
            tmp = tempfile.NamedTemporaryFile(dir=dst_dir, delete=False)
            torch.save(tensor, tmp.name)
            tmp.close()
            os.replace(tmp.name, out_path)  # atomic move
            return True
    except Exception as e:
        print(f"[WARN] Skipping {fp.name}: {e}")
        _log_bad_fits("exception", fp, str(e), tel_idx)
        return None

def process_one(args):
    idx, pt_path, dst_dir, processor = args
    try:
        tensor = torch.load(pt_path, map_location="cpu")
        data = tensor[0].numpy()
        data = np.array(data, dtype=np.float32)
        if not np.issubdtype(data.dtype, np.floating):
            print(f"[ERROR] Data is not floating-point: {data.dtype} in {pt_path}")
            return False
        if data.size == 0 or data.ndim < 2:
            print(f"[ERROR] Empty or invalid tensor in {pt_path}")
            return False
        processed = processor.process_numpy(data, resize=False)
        # Save processed tensor
        out_path = os.path.join(dst_dir, os.path.basename(pt_path))
        torch.save(processed, out_path)
        return True
    except Exception as e:
        print(f"[WARN] Failed to process {pt_path}: {e}")
        return False


# === Mixed block planning and big binary file writer ===
def _force_chw_float32_from_pt(pt_path: Path) -> np.ndarray:
    """
    Load a .pt tensor and return a contiguous numpy array shaped (1, H, W),
    taking only the FIRST channel, in **float32**. Performs strict sanity checks.
    """
    t = torch.load(str(pt_path), map_location="cpu")
    # Normalize to CHW and keep only the first channel
    if t.ndim == 2:
        t = t.unsqueeze(0)  # (1, H, W)
    elif t.ndim == 3:
        if t.shape[0] == 1:
            pass  # already (1, H, W)
        elif t.shape[0] == 3:
            t = t[:1]  # (1, H, W)
        elif t.shape[-1] == 3:
            t = t.permute(2, 0, 1)[:1]  # HWC -> CHW, take first -> (1, H, W)
        else:
            raise ValueError(f"Unexpected tensor shape {tuple(t.shape)} in {pt_path}")
    else:
        raise ValueError(f"Unexpected tensor ndim={t.ndim} in {pt_path}")

    t = t.contiguous()
    arr = t.cpu().numpy()
    # ensure float32 for stable stats/casting later
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    if arr.shape != (1, H, W):
        raise ValueError(f"Got shape {arr.shape}, expected (1,{H},{W}) from {pt_path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {pt_path}")
    sd = float(arr.std())
    if sd <= 1e-12 or float(arr.min()) == float(arr.max()):
        raise ValueError(f"Blank (all-equal) data in {pt_path}")
    # low-std is allowed; only true blanks are rejected here

    return np.ascontiguousarray(arr, dtype=np.float32)


def iter_mixed_blocks(tel_lists: list[list[Path]], block_size: int, ratios: list[float] | None, seed: int):
    """
    Yield blocks (lists of Paths) mixed according to global ratios, with
    per-block shuffling to interleave telescopes. Stops when all lists are exhausted.
    """
    lists = [list(lst) for lst in tel_lists]
    rng = random.Random(seed)
    for lst in lists:
        rng.shuffle(lst)

    tel_count = len(lists)
    while True:
        remaining = [len(lst) for lst in lists]
        tot = sum(remaining)
        if tot == 0:
            return
        target_n = min(block_size, tot)

        # Choose targets either from global ratios or proportional to remaining.
        if ratios is not None:
            target = [r * target_n for r in ratios]
        else:
            target = [(rem / tot) * target_n for rem in remaining]

        quotas = [min(int(x), remaining[i]) for i, x in enumerate(target)]
        cur = sum(quotas)

        # Add leftover by largest fractional parts / available headroom.
        if cur < target_n:
            frac = [target[i] - quotas[i] for i in range(tel_count)]
            for i in sorted(range(tel_count), key=lambda k: frac[k], reverse=True):
                head = remaining[i] - quotas[i]
                add = min(target_n - cur, head)
                if add > 0:
                    quotas[i] += add
                    cur += add
                if cur == target_n:
                    break
        # Trim if we somehow exceeded
        elif cur > target_n:
            for i in sorted(range(tel_count), key=lambda k: quotas[k], reverse=True):
                trim = min(cur - target_n, quotas[i])
                quotas[i] -= trim
                cur -= trim
                if cur == target_n:
                    break

        # Build the pick list and shuffle inside block to interleave.
        picks = []
        for i, q in enumerate(quotas):
            picks.extend([i] * q)
        rng.shuffle(picks)

        block = []
        for i in picks:
            if lists[i]:
                block.append(lists[i].pop())
        if block:
            yield block


# === Parallel big binary writer ===
def _write_span_worker(args):
    bin_path, start_idx, span_paths, total, flush_every = args
    mm = np.memmap(bin_path, mode="r+", dtype=DTYPE, shape=(total, C, H, W))
    w = int(start_idx)
    wrote = 0
    for i, pt_path in enumerate(span_paths):
        try:
            arr32 = _force_chw_float32_from_pt(pt_path)          # validated float32 (1,H,W)
            mm[w] = arr32.astype(DTYPE, copy=False)              # cast at the last hop
        except Exception:
            mm[w] = 0  # write zeros to keep alignment
        w += 1
        wrote += 1
        if flush_every and (i % flush_every) == 0 and i > 0:
            mm.flush()
    mm.flush()
    del mm
    return wrote


def write_big_binary_from_processed_dirs_parallel(processed_dirs: list[Path],
                                                  bin_path: Path,
                                                  block_size: int = 8192,
                                                  seed: int = 12345,
                                                  workers: int = 4,
                                                  span_len: int = 16384,
                                                  flush_every: int = 0,
                                                  excluded_paths: set[str | Path] | None = None):
    """
    Parallel version of the big binary writer.

    - Preserves the SAME final on-disk order as the single-process version:
      blocks are constructed once in the main process and then concatenated.
    - Splits that global sequence into contiguous spans (length = span_len) and
      assigns each span to a worker which performs a sequential write inside its range.

    Args:
        processed_dirs : list of PT directories (per telescope)
        bin_path       : destination .bin file
        block_size     : mixing unit (same as single-process)
        seed           : mixing RNG seed
        workers        : number of writer processes
        span_len       : number of images per span (contiguous on disk)
        flush_every    : if &gt;0, call mm.flush() every this many images inside a worker
        excluded_paths : optional processed tensor paths omitted before validation and mixing
    """
    # Collect lists and compute ratios
    tel_lists = [sorted(Path(d).glob("*.pt")) for d in processed_dirs]
    if excluded_paths:
        excluded = {os.path.abspath(os.fspath(path)) for path in excluded_paths}
        before = [len(paths) for paths in tel_lists]
        tel_lists = [
            [path for path in paths if os.path.abspath(os.fspath(path)) not in excluded]
            for paths in tel_lists
        ]
        removed = [old - len(new) for old, new in zip(before, tel_lists)]
        print(f"[BIN/PAR][COORD-EXCLUDE] removed={sum(removed)} per_tel={removed}")
    raw_counts = [len(x) for x in tel_lists]
    raw_total = sum(raw_counts)
    if raw_total == 0:
        print("[BIN/PAR] No processed tensors found; skipping.")
        return

    # Pre-filter
    filtered_lists, dropped = _filter_tel_lists(
        tel_lists,
        workers=max(1, workers or 0),
        low_std_thresh=1e-3,
        blank_eps=1e-12,
        log_dir=SCRIPT_DIR
    )
    counts = [len(x) for x in filtered_lists]
    total = sum(counts)
    print(f"[BIN/PAR][FILTER] kept={total} dropped={sum(dropped)} from raw_total={raw_total} "
          f"per_tel_dropped={dropped}")
    if total == 0:
        raise RuntimeError("[BIN/PAR] After filtering, no valid tensors remain.")

    ratios = [c / total for c in counts]
    print(f"[BIN/PAR] Telescopes: counts={counts}, ratios={[round(r,4) for r in ratios]}")
    print(f"[BIN/PAR] block_size={block_size}, span_len={span_len}, workers={workers}, out={bin_path}")

    total_bytes = total * RECORD_BYTES
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bin_path.with_suffix(bin_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if bin_path.exists():
        print(f"[BIN/PAR] Will overwrite existing file after successful write: {bin_path}")
    with open(tmp_path, "wb") as f:
        f.truncate(total_bytes)

    # Plan the global mixed sequence once
    global_paths = []
    for block in iter_mixed_blocks(filtered_lists, block_size=block_size, ratios=ratios, seed=seed):
        global_paths.extend(block)
    assert len(global_paths) == total, f"Planned {len(global_paths)} != total {total}"

    # Partition into contiguous spans
    tasks = []
    start = 0
    while start < total:
        end = min(start + span_len, total)
        tasks.append((str(tmp_path), start, global_paths[start:end], total, flush_every))
        start = end

    # Dispatch
    wrote = 0
    pbar = tqdm(total=total, desc="Writing (parallel)", unit="img")
    with mp.Pool(processes=max(1, int(workers))) as pool:
        for nw in pool.imap_unordered(_write_span_worker, tasks, chunksize=1):
            wrote += int(nw)
            pbar.update(int(nw))
    pbar.close()

    # verify first record is non-zero (use fp32 for reductions to avoid fp16 overflow)
    verify_mm = np.memmap(tmp_path, mode="r", dtype=DTYPE, shape=(total, C, H, W))
    first = np.asarray(verify_mm[0], dtype=np.float32)
    nz = int(np.count_nonzero(first))
    vmin, vmax = float(np.min(first)), float(np.max(first))
    vstd = float(np.std(first, dtype=np.float32))
    print(f"[BIN/PAR][VERIFY] first_rec nnz={nz} min={vmin:.4g} max={vmax:.4g} std={vstd:.4g} (fp32 stats)")
    del verify_mm
    os.replace(tmp_path, bin_path)
    gib = total_bytes / (1 << 30)
    print(f"[BIN/PAR] Done: {wrote} / {total} images → {gib:.2f} GiB at {bin_path}")


 # === Debug helper: inspect a few indices across RAW and PROCESSED ===
def debug_inspect_tensors(indices,
                          raw_dirs: list[Path] = DST_RAW,
                          proc_dirs: list[Path] = DST_PROCESSED,
                          out_path: Path | str = SCRIPT_DIR / "debug_inspect.png",
                          print_stats: bool = True):
    """
    For each telescope pair (raw_dir, proc_dir), try to load tensors with the given
    0-based integer indices (filenames like 00000042.pt), and visualize them side-by-side
    (RAW | PROCESSED). Also prints per-image min/max to stdout.

    Visualization uses simple full-range min–max scaling per image for display only.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[DEBUG] matplotlib not available; skipping debug plots.")
        plt = None

    # Collect (tel_name, idx, raw_path, proc_path, raw_array, proc_array, raw_min, raw_max, proc_min, proc_max)
    collected = []

    def _load_first_channel(pt: Path) -> np.ndarray:
        t = torch.load(str(pt), map_location="cpu")
        if isinstance(t, torch.Tensor):
            if t.ndim == 2:
                arr = t.cpu().numpy().astype(np.float32)
            elif t.ndim == 3:
                arr = t[0].cpu().numpy().astype(np.float32)
            else:
                raise ValueError(f"unexpected tensor ndim={t.ndim} in {pt}")
        else:
            arr = np.array(t, dtype=np.float32)
        return arr

    # Iterate telescopes
    for raw_dir, proc_dir in zip(raw_dirs, proc_dirs):
        tel_name = Path(raw_dir).name
        for idx in indices:
            raw_p = Path(raw_dir) / f"{int(idx):08d}.pt"
            proc_p = Path(proc_dir) / f"{int(idx):08d}.pt"

            raw_arr = None
            proc_arr = None
            if raw_p.exists():
                try:
                    raw_arr = _load_first_channel(raw_p)
                except Exception as e:
                    print(f"[DEBUG] Failed to load RAW {raw_p}: {e}")
            else:
                print(f"[DEBUG] RAW missing: {raw_p}")

            if proc_p.exists():
                try:
                    proc_arr = _load_first_channel(proc_p)
                except Exception as e:
                    print(f"[DEBUG] Failed to load PROC {proc_p}: {e}")
            else:
                print(f"[DEBUG] PROC missing: {proc_p}")

            if raw_arr is None and proc_arr is None:
                continue

            raw_min = float(np.min(raw_arr)) if raw_arr is not None else float("nan")
            raw_max = float(np.max(raw_arr)) if raw_arr is not None else float("nan")
            proc_min = float(np.min(proc_arr)) if proc_arr is not None else float("nan")
            proc_max = float(np.max(proc_arr)) if proc_arr is not None else float("nan")

            collected.append((tel_name, int(idx), raw_p, proc_p, raw_arr, proc_arr, raw_min, raw_max, proc_min, proc_max))

            if print_stats:
                print(f"[DEBUG][{tel_name}][idx={int(idx):08d}] RAW min={raw_min:.6g} max={raw_max:.6g} | PROC min={proc_min:.6g} max={proc_max:.6g}")

    if not collected or plt is None:
        return

    # Build figure: one row per (tel, idx); 2 columns (RAW | PROC)
    rows = len(collected)
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, max(1, rows) * 3.2))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for r, (tel_name, idx, raw_p, proc_p, raw_arr, proc_arr, raw_min, raw_max, proc_min, proc_max) in enumerate(collected):
        ax0, ax1 = axes[r, 0], axes[r, 1]
        # RAW
        ax0.axis("off")
        if raw_arr is not None:
            lo, hi = float(raw_arr.min()), float(raw_arr.max())
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            img = np.clip((raw_arr - lo) / denom, 0, 1)
            ax0.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax0.set_title(f"RAW {tel_name}\n{idx:08d} | min={raw_min:.3g} max={raw_max:.3g}", fontsize=8)
        else:
            ax0.text(0.5, 0.5, "RAW MISSING", ha="center", va="center", fontsize=9)
        # PROC
        ax1.axis("off")
        if proc_arr is not None:
            lo, hi = float(proc_arr.min()), float(proc_arr.max())
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            img = np.clip((proc_arr - lo) / denom, 0, 1)
            ax1.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax1.set_title(f"PROC {Path(proc_p).parent.name}\n{idx:08d} | min={proc_min:.3g} max={proc_max:.3g}", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "PROC MISSING", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DEBUG] Saved debug side-by-side plot to {out_path}")


if __name__ == "__main__":
    workers = 32 #max(1, (os.cpu_count() or 4) - 1) #16
    chunksz = 1024  # Tune for your I/O pattern
    processor = RadioFoundationProcessor(image_size=512)

    # print("\nStarting tensors creation …")
    # succeeded = 0
    # total = 0
    # for src_path, dst_raw in zip(SRC, DST_RAW):
    #     paths = sorted(src_path.rglob("*.fits"))
    #     if not paths:
    #         print(f"[WARN] No FITS files found in {src_path}")
    #         continue
    #     args = [(idx, fp, dst_raw) for idx, fp in enumerate(paths)]
    #     total += len(args)
    #     with mp.Pool(processes=workers) as pool:
    #         for result in tqdm(pool.imap_unordered(convert_one, args, chunksize=chunksz),
    #                            total=len(args),
    #                            desc=f"Converting {src_path.name} ({workers}×{chunksz})"):
    #             if result is not None:
    #                 succeeded += 1
    # print(f"Done: {succeeded}/{total} tensors written; {total-succeeded} skipped.")
    # summarize_bad_fits_logs(SCRIPT_DIR)

    print("\nStarting tensors normalization …")
    processed_count = 0
    total_files = 0
    for dst_raw, dst_proc in zip(DST_RAW, DST_PROCESSED):
        pt_files = sorted(dst_raw.glob("*.pt"))
        if not pt_files:
            print(f"[WARN] No raw tensors in {dst_raw}; skipping.")
            continue
        args_list = [(i, str(pt), str(dst_proc), processor) for i, pt in enumerate(pt_files)]
        total_files += len(args_list)
        with mp.Pool(processes=workers) as pool:
            for result in tqdm(pool.imap_unordered(process_one, args_list),
                               total=len(args_list),
                               desc=f"Processing {dst_raw.name} ({workers} workers)"):
                processed_count += int(result)
    print(f"Processed {processed_count} / {total_files} files.")

    _DBG_IDXS = [0, 1, 2, 7, 1000]
    debug_inspect_tensors(_DBG_IDXS)

    print("\nStarting binary file creation …")
    # Build a single large .bin with block-wise mixed ordering across telescopes.
    # Tune block_size for locality (larger => more sequential I/O; 8192–65536 is typical).
    BIN_BLOCK_SIZE = 8192
    BIN_SEED = 12345
    
    # Parallel writer: tune workers (2-8 is usually ideal on fast NVMe), and span_len (e.g., 8k–32k)
    write_big_binary_from_processed_dirs_parallel(DST_PROCESSED, BIN_FILE,
                                                  block_size=BIN_BLOCK_SIZE,
                                                  seed=BIN_SEED,
                                                  workers=16,
                                                  span_len=BIN_BLOCK_SIZE*2,
                                                  flush_every=0)
