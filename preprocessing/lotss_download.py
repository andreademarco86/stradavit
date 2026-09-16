#!/usr/bin/env python3
"""
Download LoTSS DR2 high-res FITS cutouts for sources in Horton+ 2025 table,
and write one JSON sidecar per FITS with multi-label classifications.

Inputs:
- catalog.txt : table (as provided)

Outputs:
- one FITS file per source in --outdir
- one JSON file per FITS in --outdir (same basename, .json)

Requires:
pip install pandas numpy requests astropy
"""

import argparse
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

CUTOUT_URL_HIRES = "https://lofar-surveys.org/dr2-cutout.fits"

# Name format: ILTJHHMMSS.ss±DDMMSS.s
PAT_ILTJ = re.compile(
    r"^ILTJ"
    r"(?P<hh>\d{2})(?P<mm>\d{2})(?P<ss>\d{2}\.\d+)"
    r"(?P<sign>[+-])"
    r"(?P<dd>\d{2})(?P<dm>\d{2})(?P<ds>\d{2}\.\d+)$"
)

# Fixed-width slices for minimal columns (Name, zbest, Size_kpc)
COLSPECS = [
    (0, 22),   # Name
    (24, 47),  # zbest
    (48, 71),  # Size (kpc)
]
COLNAMES = ["Name", "zbest", "Size_kpc"]

# Taxonomy (codes + text), derived from boolean flags
INITIAL_ORDER = [
    ("fri", "FRI", "INIT-1"),
    ("frii", "FRII", "INIT-2"),
    ("hybrid", "Hybrid", "INIT-3"),
    ("spiral", "Spiral", "INIT-4"),
    ("relaxed", "Relaxed double", "INIT-5"),
]
MORPH_ORDER = [
    ("cshaped", "C-curvature", "MORPH-1"),
    ("sshaped", "S-curvature", "MORPH-2"),
    ("misaligned", "Misalignment", "MORPH-3"),
    ("wings", "Wings", "MORPH-4"),
    ("xshaped", "X-shaped", "MORPH-5"),
    ("straight", "Straight jets", "MORPH-6"),
    ("multihotspots", "Multiple hotspots", "MORPH-7"),
    ("continuous", "Continuous jets", "MORPH-8"),
    ("banding", "Banding", "MORPH-9"),
    ("onesided", "One-sided", "MORPH-10"),
    ("restarted", "Restarted", "MORPH-11"),
]
ENV_ORDER = [
    ("cluster", "Cluster"),
    ("merger", "Merger"),
    ("diffuse", "Diffuse emission"),
    ("unknown", "Unknown"),
]

def iltj_to_radec_sexagesimal(name: str) -> tuple[str, str]:
    m = PAT_ILTJ.match(name.strip())
    if not m:
        raise ValueError(f"Unparsable ILTJ name: {name!r}")
    ra = f"{m['hh']}:{m['mm']}:{m['ss']}"
    dec = f"{m['sign']}{m['dd']}:{m['dm']}:{m['ds']}"
    return ra, dec

def size_arcmin_from_kpc(zbest: float, size_kpc: float) -> float:
    """Angular largest extent in arcmin from physical size (kpc) and redshift."""
    DA_kpc = cosmo.angular_diameter_distance(zbest).to(u.kpc).value
    theta_rad = size_kpc / DA_kpc
    theta_arcmin = theta_rad * (180.0 / np.pi) * 60.0
    return float(theta_arcmin)

def compute_cutout_size_arcmin(
    zbest: float | None,
    size_kpc: float | None,
    factor: float,
    default_arcmin: float,
) -> float | None:
    """
    Compute cutout angular size in arcmin from catalog columns.

    No minimum/maximum clamping. If inputs are missing/invalid, use default_arcmin.
    """
    if (not np.isfinite(default_arcmin)) or default_arcmin <= 0:
        raise ValueError(f"default_arcmin must be finite and > 0 (got {default_arcmin!r})")

    if zbest is None or size_kpc is None:
        size_arcmin = float(default_arcmin)
    elif (not np.isfinite(zbest)) or zbest <= 0:
        size_arcmin = float(default_arcmin)
    elif (not np.isfinite(size_kpc)) or size_kpc <= 0:
        size_arcmin = float(default_arcmin)
    else:
        size_arcmin = size_arcmin_from_kpc(zbest, size_kpc)

    if factor > 1.0:
        size_arcmin *= factor
    if (not np.isfinite(size_arcmin)) or size_arcmin <= 0:
        return None
    return float(size_arcmin)

def download_one(
    session: requests.Session,
    name: str,
    ra: str,
    dec: str,
    size_arcmin: float,
    outpath: pathlib.Path,
    timeout: float,
) -> None:
    params = {"pos": f"{ra} {dec}", "size": f"{size_arcmin:.4f}"}
    r = session.get(CUTOUT_URL_HIRES, params=params, timeout=timeout)
    r.raise_for_status()
    outpath.write_bytes(r.content)

def parse_catalog_flags_line(line: str):
    """
    Parse one whitespace-delimited line from catalog.txt.

    We only rely on the *tail* layout:
      - last 9 numbers: FeatureCT, Hasany1, hasany2, Hasall, Has1, Has2, SE, MS, EM
      - the 20 numbers preceding those are the booleans for:
        FRI, FRII, Hybrid, Spiral, Relaxed,
        Cshaped, Sshaped, Misalign, Wings, Xshaped, Straight, MultHSpot,
        Continuous, Banding, Onesided, Restarted,
        Cluster, Merger, Diffuse, Unknown
    """
    parts = line.strip().split()
    if not parts:
        return None

    name = parts[0]
    nums = parts[1:]
    if len(nums) < 29:
        return None

    tail9 = [int(float(x)) for x in nums[-9:]]
    prev20 = [int(float(x)) for x in nums[-29:-9]]

    initial_flags = prev20[:5]
    morph_flags = prev20[5:16]
    env_flags = prev20[16:20]

    rec = {
        "source_name": name,
        "initial": {},
        "morphology": {},
        "environment": {},
        "derived": {},
    }

    for (key, _, _), v in zip(INITIAL_ORDER, initial_flags):
        rec["initial"][key] = int(v)
    for (key, _, _), v in zip(MORPH_ORDER, morph_flags):
        rec["morphology"][key] = int(v)
    for (key, _), v in zip(ENV_ORDER, env_flags):
        rec["environment"][key] = int(v)

    # Derived / indicators
    (
        featurecount,
        hasanyone,
        hasanytwo,
        hasall,
        has_exactly_1,
        has_exactly_2,
        se,
        ms,
        em,
    ) = tail9

    rec["derived"] = {
        "featurecount": int(featurecount),
        "hasanyone": int(hasanyone),
        "hasanytwo": int(hasanytwo),
        "hasall": int(hasall),
        "has_exactly_1": int(has_exactly_1),
        "has_exactly_2": int(has_exactly_2),
        "se": int(se),
        "ms": int(ms),
        "em": int(em),
    }
    return rec

def load_flags_map(catalog_path: pathlib.Path) -> dict[str, dict]:
    flags = {}
    with catalog_path.open() as f:
        for line in f:
            rec = parse_catalog_flags_line(line)
            if rec is None:
                continue
            flags[rec["source_name"]] = rec
    return flags

def build_multilabels(rec: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    def to_list(order, flag_dict):
        out = []
        for key, text, code in order:
            if int(flag_dict.get(key, 0)) == 1:
                out.append({"code": code, "text": text})
        return out

    init = to_list(INITIAL_ORDER, rec["initial"])
    morph = to_list(MORPH_ORDER, rec["morphology"])

    env = []
    for key, text in ENV_ORDER:
        if int(rec["environment"].get(key, 0)) == 1:
            env.append({"text": text})

    derived = {
        "feature_count": rec["derived"].get("featurecount", 0),
        "has_any_one": rec["derived"].get("hasanyone", 0),
        "has_any_two": rec["derived"].get("hasanytwo", 0),
        "has_all_three": rec["derived"].get("hasall", 0),
        "has_exactly_one": rec["derived"].get("has_exactly_1", 0),
        "has_exactly_two": rec["derived"].get("has_exactly_2", 0),
        "se": rec["derived"].get("se", 0),
        "ms": rec["derived"].get("ms", 0),
        "em": rec["derived"].get("em", 0),
    }
    return init, morph, env, derived

def write_json_sidecar(fits_path: pathlib.Path, flags_rec: dict, overwrite: bool = False) -> None:
    json_path = fits_path.with_suffix(".json")
    if json_path.exists() and not overwrite:
        return

    init, morph, env, derived = build_multilabels(flags_rec)

    payload = {
        "schema": "lotss-dr2-vc-sidecar-1.0",
        "source_name": flags_rec["source_name"],
        "fits_file": fits_path.name,
        "labels": {
            "initial": init,
            "morphology": morph,
            "environment": env,
            "derived": derived,
        },
        # Keep raw flags to be lossless / regenerable
        "raw_flags": {
            **flags_rec["initial"],
            **flags_rec["morphology"],
            **flags_rec["environment"],
            **flags_rec["derived"],
        },
    }

    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(json_path)

def summarize_initial_classifications(outdir: pathlib.Path) -> None:
    """
    Summarize initial classifications from JSON sidecars already on disk.

    Counts are per-source (i.e., number of JSON files that include each label).
    """
    counts = {k: 0 for (k, _, _) in INITIAL_ORDER}
    n_json = 0
    n_bad_json = 0

    for json_path in sorted(outdir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text())
        except Exception:
            n_bad_json += 1
            continue

        n_json += 1
        raw_flags = payload.get("raw_flags")
        if isinstance(raw_flags, dict) and raw_flags:
            for key in counts:
                try:
                    val = int(raw_flags.get(key, 0))
                except Exception:
                    val = 0
                if val == 1:
                    counts[key] += 1
            continue

        labels = payload.get("labels")
        initial = labels.get("initial") if isinstance(labels, dict) else None
        codes = {d.get("code") for d in initial if isinstance(d, dict)} if isinstance(initial, list) else set()
        for key, _, code in INITIAL_ORDER:
            if code in codes:
                counts[key] += 1

    print(f"[JSON SUMMARY] files={n_json} bad_json={n_bad_json}")
    for key, text, _ in INITIAL_ORDER:
        print(f"  - {text}: {counts.get(key, 0)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog.txt", help="Path to catalog.txt")
    ap.add_argument("--outdir", default="lotss_dr2_horton_hires_cutouts", help="Output directory")

    ap.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Optional enlargement factor (>1.0 multiplies the catalog cutout size)",
    )
    ap.add_argument(
        "--default-arcmin",
        type=float,
        default=5.0,
        help="Default cutout size (arcmin) when zbest/Size_kpc missing or invalid",
    )

    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Retries per source")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests (seconds)")

    ap.add_argument("--write-json", action="store_true", default=True, help="Write JSON sidecars (default on)")
    ap.add_argument("--no-write-json", dest="write_json", action="store_false", help="Disable JSON sidecars")
    ap.add_argument("--overwrite-json", action="store_true", default=False, help="Overwrite existing JSON")

    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load minimal columns for cutout sizing
    df = pd.read_fwf(args.catalog, colspecs=COLSPECS, names=COLNAMES)

    # Load full flag map for classifications
    flags_map = load_flags_map(pathlib.Path(args.catalog))

    skipped: list[dict] = []
    defaulted: list[dict] = []
    n_ok = 0
    n_fail = 0

    with requests.Session() as session:
        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            if not name or name == "nan":
                continue

            # Some rows have blank zbest/Size; pandas reads as NaN.
            zbest = None if pd.isna(row["zbest"]) else float(row["zbest"])
            size_kpc = None if pd.isna(row["Size_kpc"]) else float(row["Size_kpc"])

            size_arcmin = compute_cutout_size_arcmin(
                zbest=zbest,
                size_kpc=size_kpc,
                factor=args.factor,
                default_arcmin=args.default_arcmin,
            )
            if size_arcmin is None:
                skipped.append(
                    {
                        "name": name,
                        "reason": "invalid computed cutout size",
                        "zbest": None if zbest is None else float(zbest),
                        "size_kpc": None if size_kpc is None else float(size_kpc),
                        "default_arcmin": float(args.default_arcmin),
                    }
                )
                continue
            if (
                zbest is None
                or size_kpc is None
                or (not np.isfinite(zbest))
                or zbest <= 0
                or (not np.isfinite(size_kpc))
                or size_kpc <= 0
            ):
                defaulted.append(
                    {
                        "name": name,
                        "reason": "missing/invalid zbest or Size_kpc; using default cutout size",
                        "zbest": None if zbest is None else float(zbest),
                        "size_kpc": None if size_kpc is None else float(size_kpc),
                        "default_arcmin": float(args.default_arcmin),
                        "size_arcmin": float(size_arcmin),
                    }
                )

            try:
                ra, dec = iltj_to_radec_sexagesimal(name)
            except Exception as e:
                skipped.append(
                    {
                        "name": name,
                        "reason": f"unparsable source name for RA/Dec: {e}",
                        "zbest": None if zbest is None else float(zbest),
                        "size_kpc": None if size_kpc is None else float(size_kpc),
                        "size_arcmin": float(size_arcmin),
                    }
                )
                continue

            outpath = outdir / f"{name}_s{size_arcmin:.2f}arcmin.fits"
            jsonpath = outpath.with_suffix(".json")

            # Resumable: if intended outputs exist, do nothing; otherwise create what's missing.
            if outpath.exists() and outpath.stat().st_size > 0:
                if args.write_json:
                    if jsonpath.exists() and jsonpath.stat().st_size > 0:
                        continue
                    rec = flags_map.get(name)
                    if rec is not None:
                        write_json_sidecar(outpath, rec, overwrite=args.overwrite_json)
                continue

            last_err = None
            for attempt in range(1, args.retries + 1):
                try:
                    download_one(session, name, ra, dec, size_arcmin, outpath, timeout=args.timeout)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if outpath.exists():
                        try:
                            outpath.unlink()
                        except OSError:
                            pass
                    time.sleep(1.0 * attempt)

            if last_err is not None:
                print(f"[FAIL] {name} size={size_arcmin:.2f} arcmin err={last_err}")
                n_fail += 1
            else:
                print(f"[OK] {name} size={size_arcmin:.2f} arcmin -> {outpath.name}")
                n_ok += 1
                if args.write_json:
                    rec = flags_map.get(name)
                    if rec is None:
                        print(f"[WARN] No flags found for {name}; JSON not written")
                    else:
                        write_json_sidecar(outpath, rec, overwrite=args.overwrite_json)

            if args.sleep > 0:
                time.sleep(args.sleep)

    if skipped:
        print(f"[SKIP] {len(skipped)} sources skipped")
        for rec in skipped:
            extra = []
            if "zbest" in rec:
                extra.append(f"zbest={rec.get('zbest')}")
            if "size_kpc" in rec:
                extra.append(f"Size_kpc={rec.get('size_kpc')}")
            if "size_arcmin" in rec:
                extra.append(f"size_arcmin={rec.get('size_arcmin')}")
            if "default_arcmin" in rec:
                extra.append(f"default_arcmin={rec.get('default_arcmin')}")
            extra_s = (" " + " ".join(extra)) if extra else ""
            print(f"  - {rec.get('name')}: {rec.get('reason')}{extra_s}")

    if defaulted:
        print(f"[DEFAULT] {len(defaulted)} sources used default cutout size")

    print(f"[SUMMARY] ok={n_ok} fail={n_fail} skip={len(skipped)} defaulted={len(defaulted)}")
    summarize_initial_classifications(outdir)

if __name__ == "__main__":
    main()
