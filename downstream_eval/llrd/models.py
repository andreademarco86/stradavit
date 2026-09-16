import os
import re

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from transformers import AutoConfig, Trainer
from transformers.modeling_outputs import ImageClassifierOutput

from pretraining.vit_utils import (
    get_hf_dinov2_model_cls,
    load_hf_dinov2_config,
)
from downstream_eval.llrd.runtime import _is_main_process, guarded_print

HEAD_LR_MULT = 10.0
LAYER_DECAY = 0.65
DEBUG_LAYER_DECAY = 1


class DifferentialLRTrainer(Trainer):
    """Trainer with MAE-style layer-wise decay + head multiplier."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None and hasattr(model, "strada_class_weights"):
            weight = getattr(model, "strada_class_weights")
            loss = nn.CrossEntropyLoss(weight=weight.to(outputs.logits.device))(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
            )
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _extract_layer_index(name: str):
        patterns = [
            r"(?:^|\.)(?:encoder\.)?layer\.(\d+)\.",  # vit.encoder.layer.0 / encoder.layer.0
            r"(?:^|\.)(?:encoder\.)?layers\.(\d+)\.",  # encoder.layers.0
            r"(?:^|\.)(?:blocks|block)\.(\d+)\.",  # blocks.0 / block.0
            r"(?:^|\.)(?:layers)\.(\d+)\.",  # layers.0
        ]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return int(m.group(1))
        return None

    def _infer_num_layers(self):
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for key in ("num_hidden_layers", "num_layers", "n_layers"):
                val = getattr(cfg, key, None)
                if isinstance(val, int):
                    return val
        max_idx = -1
        for name, _ in self.model.named_parameters():
            idx = self._extract_layer_index(name)
            if idx is not None:
                max_idx = max(max_idx, idx)
        return max_idx + 1 if max_idx >= 0 else 0

    def _get_layer_id(self, name: str, num_layers: int) -> int:
        head_prefixes = ("classifier", "pre_logits", "norm", "fc_norm", "head")
        backbone_prefixes = ("vit", "dino", "dinov2", "backbone", "encoder", "model")
        is_head_prefix = name in head_prefixes or name.startswith(tuple(p + "." for p in head_prefixes))
        is_backbone_root = name.startswith(tuple(p + "." for p in backbone_prefixes))
        if is_head_prefix and not is_backbone_root:
            return num_layers
        if "embeddings" in name or ".embed." in name or "patch_embed" in name:
            return 0
        idx = self._extract_layer_index(name)
        if idx is not None:
            return idx
        return 0

    def create_optimizer(self):
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer

        num_layers = self._infer_num_layers()
        decay = self.args.weight_decay
        layer_decay = LAYER_DECAY
        base_lr = self.args.learning_rate
        head_lr = base_lr * HEAD_LR_MULT

        param_groups = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            layer_id = self._get_layer_id(name, num_layers)
            is_head = layer_id == num_layers
            lr = head_lr if is_head else base_lr * (layer_decay ** max(0, (num_layers - 1 - layer_id)))
            no_wd = (
                    "bias" in name
                    or "norm" in name
                    or "position_embeddings" in name
                    or "cls_token" in name
            )
            wd = 0.0 if no_wd else decay
            key = (lr, wd)
            param_groups.setdefault(key, []).append(param)

        optimizer_grouped_parameters = [
            {"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in param_groups.items()
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        if DEBUG_LAYER_DECAY:
            samples = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                layer_id = self._get_layer_id(name, num_layers)
                is_head = layer_id == num_layers
                lr = head_lr if is_head else base_lr * (layer_decay ** max(0, (num_layers - 1 - layer_id)))
                no_wd = (
                        "bias" in name
                        or "norm" in name
                        or "position_embeddings" in name
                        or "cls_token" in name
                )
                wd = 0.0 if no_wd else decay
                samples.append((layer_id, lr, wd, name))
            samples.sort(key=lambda x: (x[0], x[3]))
            guarded_print("[LayerDecay] sample param -> layer_id / lr / wd:")
            head_samples = [s for s in samples if s[0] == num_layers]
            early = samples[:25]
            late = samples[-25:]
            to_print = early + late
            if head_samples:
                to_print += head_samples[:10]
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for s in to_print:
                key = (s[0], s[3])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(s)
            for layer_id, lr, wd, name in unique:
                guarded_print(f"  L{layer_id:02d} lr={lr:.2e} wd={wd:.2e} {name}")

        try:
            total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            guarded_print(
                f"[LayerDecay] lr={base_lr} head_mult={HEAD_LR_MULT} decay={layer_decay} "
                f"groups={len(optimizer_grouped_parameters)} trainable={total_trainable:,}"
            )
        except Exception:
            pass

        return self.optimizer


def _candidate_dirs(path: str) -> list[str]:
    cands = []
    cur = os.path.abspath(path)
    for _ in range(4):
        if cur not in cands:
            cands.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return cands


def _resolve_config_source(checkpoint_path: str) -> str:
    for cand in _candidate_dirs(checkpoint_path):
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return checkpoint_path


def _load_state_dict_from_dir(checkpoint_path: str) -> dict[str, torch.Tensor] | None:
    if not os.path.isdir(checkpoint_path):
        return None
    safe_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(safe_path):
        return safe_load_file(safe_path)
    if os.path.isfile(bin_path):
        try:
            return torch.load(bin_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(bin_path, map_location="cpu")
    return None


class DINOv2PoolerClassifier(nn.Module):
    """Paper-parity DINO classifier: native HF backbone pooler + Linear head."""

    @staticmethod
    def _load_backbone(checkpoint_path: str):
        state = _load_state_dict_from_dir(checkpoint_path)
        if state is None:
            cfg = load_hf_dinov2_config(checkpoint_path)
            model_cls = get_hf_dinov2_model_cls(cfg)
            return model_cls.from_pretrained(checkpoint_path, config=cfg)

        config_source = _resolve_config_source(checkpoint_path)
        cfg = load_hf_dinov2_config(config_source)
        model_cls = get_hf_dinov2_model_cls(cfg)
        backbone = model_cls(cfg)

        backbone_state = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("vit."):
                backbone_state[key[len("vit."):]] = value
            elif key.startswith("dino."):
                backbone_state[key[len("dino."):]] = value
            elif key.startswith("dinov2."):
                backbone_state[key[len("dinov2."):]] = value
            elif key.startswith("dinov2_with_registers."):
                backbone_state[key[len("dinov2_with_registers."):]] = value
            elif key.startswith("classifier.") or key.startswith("projector_head."):
                continue
            else:
                backbone_state[key] = value

        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if _is_main_process():
            if missing:
                guarded_print(f"   → DINO local ckpt missing keys (truncated): {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                guarded_print(f"   → DINO local ckpt unexpected keys (truncated): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        return backbone

    def __init__(self, checkpoint_path: str, num_labels: int, class_weights=None):
        super().__init__()
        self.dino = self._load_backbone(checkpoint_path)
        self.config = self.dino.config
        self.num_labels = int(num_labels)
        hidden_size = int(getattr(self.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("Could not infer DINO hidden_size for pooler classifier.")

        guarded_print(
            f"   → DINO backbone class: {self.dino.__class__.__module__}.{self.dino.__class__.__name__} "
            f"(model_type={getattr(self.config, 'model_type', None)!r}, "
            f"num_register_tokens={getattr(self.config, 'num_register_tokens', None)!r})"
        )
        guarded_print("   → DINO readout: backbone pooler_output -> Linear(hidden_size, classes)")

        dropout_prob = float(getattr(self.config, "classifier_dropout_prob", 0.0) or 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pixel_values=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.dino(pixel_values=pixel_values, **kwargs)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            raise RuntimeError(
                f"{self.dino.__class__.__name__} did not return pooler_output. "
                "DINO evaluation requires the paper-parity pooler readout."
            )
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            if getattr(self, "class_weights", None) is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def apply_strict_linear_probe_head(model: nn.Module, num_classes: int):
    """Replace model head with Linear-only and freeze everything else."""
    hidden_size = None
    cfg = getattr(model, "config", None)
    if cfg is not None:
        hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None and hasattr(model, "classifier"):
        hidden_size = getattr(model.classifier, "in_features", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for strict linear probe head.")

    # Replace common head modules with Identity for strict linear probe
    if hasattr(model, "norm"):
        model.norm = nn.Identity()
    if hasattr(model, "fc_norm"):
        model.fc_norm = nn.Identity()
    if hasattr(model, "pre_logits"):
        model.pre_logits = nn.Identity()
    if hasattr(model, "dropout"):
        model.dropout = nn.Identity()

    model.classifier = nn.Linear(hidden_size, num_classes)
    nn.init.trunc_normal_(model.classifier.weight, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    # Freeze everything, then unfreeze classifier only
    for _, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def compute_metrics(eval_pred):
    """Compute macro/weighted F1 for Trainer eval/predict calls."""
    from sklearn.metrics import f1_score as _f1

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    macro_f1 = _f1(labels, preds, average="macro", zero_division=0)
    weighted_f1 = _f1(labels, preds, average="weighted", zero_division=0)
    return {"macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}


def _weighted_f1_from_cm(cm: np.ndarray) -> float:
    if cm.size == 0:
        return 0.0
    support = cm.sum(axis=1)
    total = support.sum()
    if total == 0:
        return 0.0
    per_class_f1 = []
    for cls_idx in range(cm.shape[0]):
        tp = cm[cls_idx, cls_idx]
        fp = cm[:, cls_idx].sum() - tp
        fn = cm[cls_idx, :].sum() - tp
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1_i = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_class_f1.append(f1_i)
    per_class_f1 = np.array(per_class_f1, dtype=np.float32)
    return float((per_class_f1 * support).sum() / total)


def infer_n_registers(model_id: str, encoder_family: str) -> int:
    """Infer number of register tokens from config or run name."""
    if encoder_family != "mae":
        return 0
    regs = 0
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        regs = int(getattr(cfg, "n_registers", 0) or 0)
    except Exception:
        regs = 0
    if regs == 0:
        m = re.search(r"regs=([0-9]+)", model_id)
        if m:
            regs = int(m.group(1))
    return regs


def _pretrained_candidate_dirs(path: str) -> list[str]:
    if not isinstance(path, str) or not path:
        return []
    cur = os.path.abspath(path)
    out = []
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def _resolve_pretrained_config_source(path: str) -> str:
    """Return the nearest local directory containing config.json, else the original path."""
    for cand in _pretrained_candidate_dirs(path):
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return path


def _extract_image_size_from_processor(proc) -> int | None:
    if proc is None:
        return None

    candidates = []
    for attr in ("size", "crop_size"):
        val = getattr(proc, attr, None)
        if val is not None:
            candidates.append(val)

    for val in candidates:
        if isinstance(val, int):
            return int(val)
        if isinstance(val, dict):
            for key in ("height", "width", "shortest_edge"):
                maybe = val.get(key)
                if maybe is not None:
                    try:
                        return int(maybe)
                    except Exception:
                        continue
    return None


def load_hf_dinov2_config_dict(model_name_or_path: str) -> dict:
    """Read raw DINO config.json without depending on vit_utils helper availability."""
    try:
        from transformers.configuration_utils import PretrainedConfig  # type: ignore
        cfg_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
        return cfg_dict if isinstance(cfg_dict, dict) else {}
    except Exception:
        return {}
