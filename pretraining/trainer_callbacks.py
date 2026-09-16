import sys
sys.path.append('../')
# --- make local packages importable regardless of CWD ---
import os, sys, random, datetime, math
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----- rank-aware printing helpers -----
def _is_main_process():
    return os.getenv("RANK", "0") in ("0", "") or os.getenv("LOCAL_RANK", "0") in ("0", "")

def guarded_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)

from transformers import TrainerCallback
import torch
# Prefer TF32 on Ampere/Ada for faster GEMMs while retaining fp32 API numerics
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from torchvision.utils import save_image


class ReconVizCallback(TrainerCallback):
    """
    Save reconstruction diagnostics at epoch-end.

    For each group, saves four rows:
      1) original image
      2) original image with MAE-masked patches shown as black
      3) decoder reconstruction
      4) residuals (absolute error * err_gain)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    processor : ViTImageProcessor (for un-normalising)
    out_dir : str
    num_samples : int
        Total images to save (must be a multiple of group_size).
    group_size : int
        How many images per row (max 8).
    vary_samples : bool
        If True, choose a new random set every epoch; otherwise
        reuse the same indices.
    err_gain : float
        Gain to apply to residual error images.
    """
    def __init__(self, dataset, processor, out_dir,
                 num_samples=24, group_size=8, vary_samples=False, err_gain=1.0):
        assert num_samples % group_size == 0, \
            "num_samples must be a multiple of group_size"
        assert group_size <= 8, "group_size must be ≤ 8"
        self.dataset = dataset
        self.processor = processor
        self.out_dir = out_dir
        self.num_samples = num_samples
        self.group_size = group_size
        self.vary = vary_samples
        self.err_gain = err_gain
        os.makedirs(out_dir, exist_ok=True)

        # Fixed indices for deterministic mode
        self.indices = list(range(len(dataset)))
        self._last_saved_epoch_idx = None

    @staticmethod
    def _patch_mask_to_image_mask(mask: torch.Tensor | None, image_shape: torch.Size) -> torch.Tensor | None:
        """Expand a (B, num_patches) MAE mask to (B, 1, H, W)."""
        if mask is None:
            return None
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.dim() != 2:
            return None

        b, _, h, w = image_shape
        if int(mask.shape[0]) != int(b):
            return None

        n_patches = int(mask.shape[1])
        grid = int(math.sqrt(n_patches))
        if grid * grid != n_patches or h % grid != 0 or w % grid != 0:
            return None

        mask_bool = mask if mask.dtype == torch.bool else mask > 0.5
        patch_h = h // grid
        patch_w = w // grid
        return (
            mask_bool.reshape(b, grid, grid)
            .repeat_interleave(patch_h, dim=1)
            .repeat_interleave(patch_w, dim=2)
            .unsqueeze(1)
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        # HF Trainer keeps `state.epoch` as a float, typically reaching an integer at epoch end
        # (e.g. 1.0 at the end of epoch 0). Convert to a 0-based epoch index robustly and
        # avoid saving twice if this callback fires multiple times for the same epoch.
        epoch_idx = 0
        try:
            if state.epoch is None:
                epoch_idx = 0 if self._last_saved_epoch_idx is None else int(self._last_saved_epoch_idx) + 1
            else:
                # At epoch end, `state.epoch` is often an integer boundary for the *next* epoch.
                # `ceil(x - eps) - 1` maps 1.0 -> 0, 2.0 -> 1, while keeping 0.999.. -> 0.
                epoch_idx = max(0, int(math.ceil(float(state.epoch) - 1e-12) - 1))
        except Exception:
            epoch_idx = 0

        if self._last_saved_epoch_idx == epoch_idx:
            return
        self._last_saved_epoch_idx = epoch_idx

        model = kwargs["model"].eval()
        device = next(model.parameters()).device

        # Choose indices for this epoch
        total = len(self.dataset)
        nsamp = min(self.num_samples, total)
        if self.vary:
            sel = random.sample(range(total), nsamp)
        else:
            sel = list(range(nsamp))

        # Batch images, handling multi-view samples (e.g., FULL/BYOL mode)
        def _first_view(sample):
            pv = sample["pixel_values"]
            if isinstance(pv, (list, tuple)):
                return pv[0]  # use v1 for reconstruction visualization
            return pv

        imgs = torch.stack([_first_view(self.dataset[i]) for i in sel]).to(device)

        # Ensure channel dimension matches model config (expand 1->3 for grayscale)
        if imgs.dim() == 4 and imgs.size(1) == 1:
            imgs = imgs.expand(-1, 3, -1, -1).contiguous()

        mean = torch.tensor(self.processor.image_mean).view(-1, 1, 1).to(device)
        std = torch.tensor(self.processor.image_std).view(-1, 1, 1).to(device)

        with torch.no_grad():
            out = model(pixel_values=imgs)
            recon = model.unpatchify(out.logits)
            recon = recon * std + mean
            recon = recon.clamp(0, 1)

        orig = imgs * std + mean
        orig = orig.clamp(0, 1)

        masked_input = orig.clone()
        image_mask = self._patch_mask_to_image_mask(getattr(out, "mask", None), orig.shape)
        if image_mask is not None:
            # Visualization only: the encoder receives visible patch tokens,
            # not a literal black-patched image.
            masked_input = masked_input.masked_fill(image_mask.to(device=masked_input.device), 0.0)
        else:
            guarded_print("[ReconViz] model output did not expose a compatible patch mask; masked-input row equals original row.")

        resid = (orig - recon).abs() * self.err_gain
        resid = resid.clamp(0, 1)

        # Save all groups in a single file with alternating 4-row blocks.
        g = min(self.group_size, 8)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        grids = []
        for k in range(0, nsamp, g):
            o = orig[k:k+g]
            m = masked_input[k:k+g]
            r = recon[k:k+g]
            e = resid[k:k+g]
            grids.append(torch.cat([o, m, r, e], dim=0))  # 4 * g images per group

        big_grid = torch.cat(grids, dim=0)  # (4*g*ngroups, C, H, W)
        save_path = os.path.join(
            self.out_dir,
            f"epoch{epoch_idx}_all_{now}.png"
        )
        save_image(big_grid, save_path, nrow=g)
        guarded_print(f"[ReconViz] saved combined grid: {save_path} (rows = {4 * (nsamp // g)}; original, masked_input, recon, error)")

# --- MetricsPlotCallback: Save metrics summary plot and CSV at train end ---
class MetricsPlotCallback(TrainerCallback):
    """
    At train end, aggregate Trainer.state.log_history by epoch and save:
      - metrics_summary.png with key curves vs epoch
    Files go into the parent run folder (the same folder you pass as `output_dir`).
    Plots include a shaded ±1σ band around the epoch means.
    """
    def __init__(self, out_dir: str, keys: list[str] | None = None, filename: str = "metrics_summary.png"):
        self.out_dir = out_dir
        self.keys = keys
        self.filename = filename

    def on_train_end(self, args, state, control, **kwargs):
        import os, math, numpy as np, numbers
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from collections import defaultdict

        logs = state.log_history or []
        if not logs:
            guarded_print("[MetricsPlot] No log_history found; skipping.")
            return

        epoch_scale = 1.0
        if str(getattr(args, "epoch_semantics", "")) == "mega_chunk_local":
            try:
                mega_chunk_count = int(getattr(args, "mega_chunk_count", 1) or 1)
            except Exception:
                mega_chunk_count = 1
            epoch_scale = 1.0 / float(max(1, mega_chunk_count))

        # bucket numeric metrics by epoch
        def _to_float(v):
            # Accept python scalars, numpy scalars, and 0-d torch tensors
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, numbers.Number):
                try:
                    return float(v)
                except Exception:
                    return None
            try:
                import torch as _torch
                if isinstance(v, _torch.Tensor):
                    if v.numel() != 1:
                        return None
                    return float(v.detach().cpu().item())
            except Exception:
                pass
            try:
                # 0-d numpy arrays or array-likes
                if hasattr(v, "shape") and getattr(v, "shape", None) == ():
                    return float(v.item())
            except Exception:
                pass
            return None

        buckets = defaultdict(lambda: defaultdict(list))
        for rec in logs:
            ep = rec.get("epoch", None)
            if ep is None:
                continue
            try:
                ep = float(ep)
            except Exception:
                continue
            ep *= epoch_scale
            for k, v in rec.items():
                if k in ("epoch", "step"):
                    continue
                fv = _to_float(v)
                if fv is not None:
                    buckets[ep][k].append(fv)

        if not buckets:
            guarded_print("[MetricsPlot] No epoch-bucketed metrics; skipping.")
            return

        # choose keys:
        # - if explicit keys are provided, keep those that exist
        # - otherwise, plot all numeric metrics observed in this run
        all_keys = set()
        for d in buckets.values():
            all_keys.update(d.keys())
        excluded_keys = {
            "total_flos",
            "train_loss",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
        }
        if self.keys is not None:
            keys = [k for k in self.keys if k in all_keys and k not in excluded_keys]
        else:
            # Keep a compact, predictable ordering for the most important training signals,
            # then append all remaining metrics in lexical order.
            preferred_prefix = [
                "loss",
                "l2_loss",
                "l1_loss",
                "bl1_loss",
                "grad_norm",
                "grad_norm_preclip",
                "grad_norm_postclip",
                "grad_norm_clip_coef",
                "learning_rate",
                "feat_var",
                "n_views",
                "clr_loss",
                "soft_hcl_loss",
                "hcl_loss",
                "clr_debiased_loss",
                "clr_tau_plus",
                "soft_hcl_tau",
                "soft_hcl_alpha",
                "hcl_beta",
                "hcl_tau_plus",
                "clr_pos_sim_mean",
                "clr_neg_sim_mean",
                "clr_neg_sim_std",
                "clr_keff_est",
                "clr_cos_std",
                "clr_cos_mean_abs",
                "clr_eff_rank",
            ]
            keys = [k for k in preferred_prefix if k in all_keys]
            remaining = sorted(k for k in all_keys if k not in set(keys) and k not in excluded_keys)
            keys.extend(remaining)

        if not keys:
            guarded_print("[MetricsPlot] No requested keys present; skipping.")
            return

        # aggregate stats per epoch
        epochs = sorted(buckets.keys())
        mean_series = {k: [] for k in keys}
        std_series  = {k: [] for k in keys}
        for ep in epochs:
            for k in keys:
                vals = buckets[ep].get(k, [])
                if len(vals) == 0:
                    m = float('nan'); s = float('nan')
                elif len(vals) == 1:
                    m = float(vals[0]); s = 0.0
                else:
                    arr = np.asarray(vals, dtype=float)
                    m = float(arr.mean())
                    s = float(arr.std(ddof=0))
                mean_series[k].append(m)
                std_series[k].append(s)

        # ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # plot grid (compact)
        cols = min(3, len(keys))
        rows = math.ceil(len(keys) / cols)
        fig, axarr = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for idx, k in enumerate(keys):
            r, c = divmod(idx, cols)
            ax = axarr[r][c]
            mu = mean_series[k]
            sd = std_series[k]
            lower = [ (m - s) if (np.isfinite(m) and np.isfinite(s)) else np.nan for m, s in zip(mu, sd) ]
            upper = [ (m + s) if (np.isfinite(m) and np.isfinite(s)) else np.nan for m, s in zip(mu, sd) ]
            ax.fill_between(epochs, lower, upper, alpha=0.2, linewidth=0)
            ax.plot(epochs, mu)
            ax.set_title(k)
            ax.set_xlabel("epoch")
            if k == "grad_norm":
                try:
                    ax.set_yscale("log")
                except Exception:
                    pass
            ax.grid(True, alpha=0.3)

            # --- Overlays and twin-axes for related metrics ---
            added_legend = False

            # 1) grad_norm + learning_rate (twin y-axis)
            if k == "grad_norm" and "learning_rate" in mean_series:
                try:
                    ax.set_yscale("log")
                except Exception:
                    pass
                try:
                    ax2 = ax.twinx()
                    ax2.plot(epochs, mean_series["learning_rate"], linestyle="--", alpha=0.7, label="learning_rate")
                    ax2.set_ylabel("learning_rate")
                except Exception:
                    pass

            if added_legend:
                ax.legend(frameon=False, fontsize=8)
        # remove unused axes
        total_axes = rows * cols
        for j in range(len(keys), total_axes):
            r, c = divmod(j, cols)
            fig.delaxes(axarr[r][c])

        # Mark warmup end across all subplots if available
        try:
            warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
            warmup_end = None
            if warmup_steps > 0:
                step_epoch_pairs = []
                for rec in logs:
                    step = rec.get("step", None)
                    ep = rec.get("epoch", None)
                    try:
                        step_f = float(step)
                        ep_f = float(ep) * epoch_scale
                    except Exception:
                        continue
                    if ep_f > 0.0 and step_f >= 0.0:
                        step_epoch_pairs.append((step_f, ep_f))
                if step_epoch_pairs:
                    steps_per_epoch = max((step / ep) for step, ep in step_epoch_pairs if ep > 0.0)
                    if steps_per_epoch > 0.0:
                        warmup_end = float(warmup_steps) / float(steps_per_epoch)
            if warmup_end is None:
                configured_epochs = getattr(args, "configured_full_file_epochs", None)
                if configured_epochs is None:
                    configured_epochs = float(getattr(args, "num_train_epochs", 0)) * epoch_scale
                warmup_end = float(configured_epochs) * float(getattr(args, "warmup_ratio", 0))
            if warmup_end > 0:
                for axs in axarr.flat:
                    axs.axvline(warmup_end, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        except Exception:
            pass

        plt.tight_layout()
        png_path = os.path.join(self.out_dir, self.filename)
        try:
            fig.savefig(png_path, dpi=150)
            guarded_print(f"[MetricsPlot] saved {png_path}")
        finally:
            plt.close(fig)

# --- ProcessorSaveCallback: Save processor config alongside Trainer checkpoints ---
class ProcessorSaveCallback(TrainerCallback):
    """Ensure preprocessor_config.json is saved wherever Trainer writes checkpoints.
    - On train begin: save into args.output_dir (the "results" root).
    - On each save: also save into the current checkpoint dir.
    """
    def __init__(self, processor):
        self.processor = processor

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            self.processor.save_pretrained(args.output_dir)
            guarded_print(f"[ProcessorSave] wrote preprocessor_config.json to {args.output_dir}")
        except Exception as e:
            guarded_print(f"[ProcessorSave] on_train_begin failed: {e}")

    def on_save(self, args, state, control, **kwargs):
        try:
            # HF Trainer saves to output_dir/checkpoint-{global_step}
            if getattr(state, "global_step", None) is None:
                return
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.processor.save_pretrained(ckpt_dir)
            guarded_print(f"[ProcessorSave] wrote preprocessor_config.json to {ckpt_dir}")
        except Exception as e:
            guarded_print(f"[ProcessorSave] on_save failed: {e}")

# --- Temperature schedule callback for contrastive temperature warmup ---
class TemperatureScheduleCallback(TrainerCallback):
    def __init__(self, trainer_ref, warmup_epochs: int, temp_warm: float, temp_main: float):
        self.trainer_ref = trainer_ref
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.temp_warm = float(temp_warm)
        self.temp_main = float(temp_main)

    def _temperature_at_epoch(self, epoch: float) -> float:
        """Cosine annealing from temp_warm (start) to temp_main (end) over warmup_epochs."""
        if self.warmup_epochs <= 0:
            return self.temp_main
        t = max(0.0, min(1.0, float(epoch) / float(self.warmup_epochs)))
        import math
        # epoch=0  -> temp_warm
        # epoch>=warmup_epochs -> temp_main
        return self.temp_main + 0.5 * (self.temp_warm - self.temp_main) * (1.0 + math.cos(math.pi * t))

    def on_train_begin(self, args, state, control, **kwargs):
        const = getattr(self.trainer_ref, "force_constant_temperature", None)
        if const is not None:
            self.temp_warm = float(const)
            self.temp_main = float(const)
            self.warmup_epochs = 0
            self.trainer_ref.temperature = float(const)
            try:
                guarded_print(f"[TempSchedule:init] epoch=0 -> temperature={self.trainer_ref.temperature}")
            except Exception:
                pass
            return

        try:
            cur = float(state.epoch or 0.0)
        except Exception:
            cur = 0.0
        tau = self._temperature_at_epoch(cur)
        self.trainer_ref.temperature = tau
        try:
            guarded_print(f"[TempSchedule:init] epoch={cur:.2f} -> temperature={self.trainer_ref.temperature}")
        except Exception:
            pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        const = getattr(self.trainer_ref, "force_constant_temperature", None)
        if const is not None:
            self.trainer_ref.temperature = float(const)
            try:
                guarded_print(f"[TempSchedule] epoch={int(state.epoch or 0)} -> temperature={self.trainer_ref.temperature}")
            except Exception:
                pass
            return

        try:
            cur = float(state.epoch or 0.0)
        except Exception:
            cur = 0.0
        tau = self._temperature_at_epoch(cur)
        self.trainer_ref.temperature = tau
        try:
            guarded_print(f"[TempSchedule] epoch={cur:.2f} -> temperature={self.trainer_ref.temperature}")
        except Exception:
            pass
