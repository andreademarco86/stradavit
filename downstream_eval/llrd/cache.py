import json
import os

import downstream_eval.llrd.settings as cfg
from downstream_eval.llrd.runtime import guarded_print


def load_cache() -> dict:
    if os.path.exists(cfg.CACHE_JSON_PATH):
        with open(cfg.CACHE_JSON_PATH, "r") as f:
            cache = json.load(f)
        guarded_print(f"Loaded cached results from {cfg.CACHE_JSON_PATH}")
        return cache

    return {}


def save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(cfg.CACHE_JSON_PATH), exist_ok=True)
    with open(cfg.CACHE_JSON_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    guarded_print(f"Saved cache to {cfg.CACHE_JSON_PATH}")
