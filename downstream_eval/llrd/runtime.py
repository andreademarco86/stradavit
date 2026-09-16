import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _is_main_process():
    return os.getenv("RANK", "0") in ("0", "") or os.getenv("LOCAL_RANK", "0") in ("0", "")


def guarded_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)


def configure_runtime_environment(
    hf_cache_path: str = "./hf_cache",
    verbose: bool = False,
    *,
    deterministic_eval: bool = True,
) -> None:
    if verbose:
        guarded_print(f"[PathDebug] cwd={os.getcwd()}")
        guarded_print(f"[PathDebug] added to sys.path: {PROJECT_ROOT}")

    os.environ.update({
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "NCCL_TIMEOUT": "3600",
        "NCCL_LAUNCH_TIMEOUT": "3600",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "OMP_NUM_THREADS": "1",
    })

    if not _is_main_process():
        import logging
        import warnings

        for name in [
            "transformers",
            "transformers.trainer",
            "transformers.training_args",
            "datasets",
        ]:
            logging.getLogger(name).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = not deterministic_eval
        torch.backends.cudnn.allow_tf32 = not deterministic_eval
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest" if deterministic_eval else "high")
        if deterministic_eval:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)
        guarded_print(
            f"[Runtime] deterministic_eval={deterministic_eval} tf32={'off' if deterministic_eval else 'on'}"
        )
    except Exception as exc:
        guarded_print(f"[Determinism] Failed to enable strict deterministic evaluation: {exc}")
        raise

    os.environ["HF_HOME"] = os.path.join(hf_cache_path, "HF_HOME")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_path, "HF_CACHE")
