import os

def derive_arch_tok_from_config(cfg, src_name):
    """Return a compact arch token like 'vitmae-b' based on config or name."""
    sid = str(src_name).split('/')[-1].lower() if src_name else ""
    if 'vit' in sid and 'mae' in sid:
        if 'base' in sid:  return 'vitmae-b'
        if 'large' in sid: return 'vitmae-l'
        if 'huge' in sid:  return 'vitmae-h'
        if 'giant' in sid: return 'vitmae-g'
    hs = int(getattr(cfg, 'hidden_size', getattr(getattr(cfg, 'vit_config', cfg), 'hidden_size', 0)) or 0)
    if hs >= 1280: return 'vitmae-h'
    if hs >= 1024: return 'vitmae-l'
    if hs >=  768: return 'vitmae-b'
    return 'vitmae'

def short_arch_name(name: str) -> str:
    """Normalize common ViT-MAE names to compact tokens (e.g., 'vit-mae-base' → 'vitmae-b')."""
    s = name.lower().replace('/', '-')
    s = s.replace('vit-mae', 'vitmae')
    s = s.replace('base', 'b').replace('large', 'l').replace('huge', 'h').replace('giant', 'g')
    s = s.replace('-', '')
    for tag in ('b', 'l', 'h', 'g', 's', 't'):
        if s.endswith(tag):
            return f"{s[:-1]}-{tag}"
    return s

def fmt_lr_token(x: float) -> str:
    """Represent small LRs compactly; 0.0002 → 2e-4."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split('e')
    m = m.rstrip('0').rstrip('.')
    return f"{m}e{int(e)}"

def phase_tag_from_output_dir(output_dir: str) -> str | None:
    """Return 'phase1'/'phase2' if output_dir ends with those markers."""
    try:
        base_name = os.path.basename(output_dir.rstrip('/'))
        if base_name in ("phase1", "phase2"):
            return base_name
    except Exception:
        return None
    return None
