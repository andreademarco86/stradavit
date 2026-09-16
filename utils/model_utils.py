import os
import sys
import math
import random
from subprocess import call

import numpy as np
import torch

try:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel, logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    init_empty_weights = None
    AutoConfig = None
    AutoModel = None

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

def set_all_seeds(seed: int = 42):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_after_run(logger=None, scoped_objects=None):
    """Best-effort cleanup to release this run's DataLoader workers and CUDA memory between runs."""
    n_iters_shutdown = 0
    n_loaders_shutdown = 0
    n_iters_cleared = 0

    try:
        import gc
        seen = set()

        def _shutdown_worker_iter(it):
            nonlocal n_iters_shutdown
            if it is None:
                return
            try:
                shutdown = getattr(it, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
                    n_iters_shutdown += 1
            except Exception:
                pass

        def _cleanup_scoped_object(obj):
            nonlocal n_loaders_shutdown, n_iters_cleared
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in seen:
                return
            seen.add(obj_id)

            _shutdown_worker_iter(obj)

            try:
                it = getattr(obj, "_iterator", None)
            except Exception:
                it = None
            if it is not None:
                _shutdown_worker_iter(it)
                n_loaders_shutdown += 1
                try:
                    obj._iterator = None
                    n_iters_cleared += 1
                except Exception:
                    pass

            for attr in (
                "_train_dataloader",
                "train_dataloader",
                "_eval_dataloader",
                "eval_dataloader",
                "_test_dataloader",
                "test_dataloader",
                "dataloader",
                "base_dataloader",
            ):
                try:
                    child = getattr(obj, attr, None)
                except Exception:
                    child = None
                if child is not None and child is not obj:
                    _cleanup_scoped_object(child)

        for obj in tuple(scoped_objects or ()):
            _cleanup_scoped_object(obj)

        # Collect after worker shutdown to free unreferenced objects/cycles.
        gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            if logger is not None:
                try:
                    logger(
                        "[Cleanup] shutdown_iters="
                        f"{n_iters_shutdown} shutdown_loaders={n_loaders_shutdown} "
                        f"cleared_loader_iters={n_iters_cleared} + cuda.empty_cache() + cuda.ipc_collect()"
                    )
                except Exception:
                    pass
    except Exception as e:
        if logger is not None:
            try:
                logger(f"[Cleanup] failed: {e}")
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# GPU / device utilities (merged from gpu_utils.py)
# ──────────────────────────────────────────────────────────────

def estimate_model_size_in_bytes(model_name: str, dtype: torch.dtype, model_cls=None, **from_pretrained_kwargs) -> int:
    """
    Instantiate a *skeleton* of your model on the 'meta' device to count params,
    then estimate the bytes needed for weights + grads + optimizer states (overhead factor).
    """
    model_cls = model_cls or AutoModel
    if init_empty_weights is None or AutoConfig is None or model_cls is None:
        raise RuntimeError("accelerate/transformers not available for size estimation")

    config = from_pretrained_kwargs.pop("config", None) \
             or AutoConfig.from_pretrained(model_name, **from_pretrained_kwargs)
    with init_empty_weights():
        constructor = getattr(model_cls, "from_config", None) or getattr(model_cls, "_from_config", None)
        if constructor is None:
            raise AttributeError(f"{model_cls.__name__} has no from_config or _from_config – cannot estimate size")
        meta_model = constructor(config)

    n_params = sum(p.numel() for p in meta_model.parameters())
    bytes_per_param = torch.finfo(dtype).bits // 8
    overhead_factor = {
        torch.float32: 3.0,
        torch.bfloat16: 2.2,
        torch.float16: 2.0,
    }.get(dtype, 3.0)

    required = n_params * bytes_per_param * overhead_factor
    required *= 1.1  # safety margin
    return int(math.ceil(required))

def gpu_min_free_bytes(margin: float = 0.9) -> int:
    """Query each CUDA device’s free memory and return the worst-case usable bytes."""
    free = []
    for i in range(torch.cuda.device_count()):
        f, _ = torch.cuda.mem_get_info(i)
        free.append(f)
    return int(min(free) * margin) if free else 0

def get_best_dtype() -> torch.dtype:
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        return torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        major = torch.cuda.get_device_properties(0).major
        if major >= 7:
            return torch.float16
        return torch.float32
    if torch.backends.mps.is_available():
        return torch.float32
    return torch.float32

def smart_load(model_name: str, dtype: torch.dtype = None, model_cls=None, **from_pretrained_kwargs):
    """
    Auto-select data vs. model parallel loading:
      • CUDA: single-GPU if it fits; else device_map='auto'.
      • MPS/CPU: direct load.
    """
    dtype     = dtype or get_best_dtype()
    model_cls = model_cls or AutoModel
    device    = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if device == "cuda":
        required  = estimate_model_size_in_bytes(model_name, dtype, model_cls=model_cls, **from_pretrained_kwargs)
        available = gpu_min_free_bytes()

        print(f"Model '{model_name}' needs ~{required/1e9:.1f} GB (incl. overhead).")
        print(f"Worst-case GPU free: {available/1e9:.1f} GB (with margin).")

        if required < available:
            print("✅ Loading onto single GPU (data parallel).")
            model = model_cls.from_pretrained(
                model_name,
                torch_dtype=dtype,
                **from_pretrained_kwargs
            ).to("cuda:0")
        else:
            print("🚨 Too big for one GPU, sharding with device_map='auto'.")
            model = model_cls.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                **from_pretrained_kwargs
            )
    else:
        print(f"ℹ️  Loading onto {device.upper()} without sharding.")
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **from_pretrained_kwargs
        ).to(device)

    return model

def get_device() -> torch.device:
    """
    Return a torch.device for this process.

    - CUDA: use LOCAL_RANK to select the correct GPU and call torch.cuda.set_device().
    - MPS: use "mps" only when CUDA is not available.
    - CPU: fallback.
    - Print environment/debug info only on global rank 0.
    """
    rank = int(os.getenv("RANK", "0"))
    local_rank_str = os.getenv("LOCAL_RANK", "0")
    try:
        local_rank = int(local_rank_str)
    except ValueError:
        local_rank = 0

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            local_rank = 0
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print('_____Python, Pytorch, Cuda info____')
            print('__Python VERSION:', sys.version)
            print('__pyTorch VERSION:', torch.__version__)
            try:
                print('__CUDNN VERSION:', torch.backends.cudnn.version())
            except Exception:
                pass
            print('_____nvidia-smi GPU details____')
            try:
                call([
                    "nvidia-smi", "--format=csv",
                    "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"
                ])
            except Exception:
                print("nvidia-smi not available.")
            print('_____Device assignments____')
            print('Number CUDA Devices:', torch.cuda.device_count())
            print('Current cuda device: ', torch.cuda.current_device(),
                  ' **May not correspond to nvidia-smi ID above, check visibility parameter')
            print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Using device: {device}")

    return device


class _DevicePrefetch:
    """
    Move the whole batch to *device* just before the forward pass.

    • CUDA  – uses a background stream so H2D copy overlaps with compute.
    • MPS/CPU – falls back to a plain `.to(device)` (no async streams).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def _move(self, obj):
        if torch.is_tensor(obj):
            if self.device.type == "cuda":
                return obj.to(self.device, non_blocking=True)
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move(v) for v in obj)
        return obj

    def __call__(self, batch: dict):
        if self.device.type == "cuda":
            with torch.cuda.stream(self.stream):
                return self._move(batch)
        return self._move(batch)
