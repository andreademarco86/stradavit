from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

from downstream_eval.llrd.config import load_config, validate_config
from downstream_eval.llrd.run_types import LLRDConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configurable Strada LLRD evaluation entrypoint.")
    parser.add_argument("--config", default=None, help="Path to a JSON/YAML LLRD evaluation config.")
    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> LLRDConfig:
    cfg = load_config(args.config)
    validate_config(cfg)
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_run_config(args)

    Path(cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
    from downstream_eval.llrd.evaluation import run_llrd_evaluation

    run_llrd_evaluation(cfg)


if __name__ == "__main__":
    main()
