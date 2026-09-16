import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from downstream_eval.llrd.settings import (
    ANALYSIS_CLASS_ORDER,
    ANALYSIS_CM_CMAP_NAME,
    ANALYSIS_CM_NORMALIZATION,
    ANALYSIS_COMPACT_TABLE_LAYOUT,
    ANALYSIS_DATASET_ORDER,
    ANALYSIS_MODELS,
    ANALYSIS_HEADER_FONT_SIZE,
    ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS,
    FIG_DPI,
    LR_CONFIGS,
    EvalMode,
    _cfg_tag,
)
from downstream_eval.llrd.runtime import SCRIPT_DIR, guarded_print

def _ordered_analysis_datasets(section_labels: list[str]) -> list[tuple[str, str]]:
    present = set(section_labels)
    ordered = [(label, display) for label, display in ANALYSIS_DATASET_ORDER if label in present]
    known = {label for label, _ in ordered}
    for label in section_labels:
        if label not in known:
            ordered.append((label, label))
    return ordered


def _analysis_cache_key(eval_mode_sel: EvalMode) -> str:
    if not LR_CONFIGS:
        raise RuntimeError("LR_CONFIGS is empty; cannot resolve analysis cache key.")
    return f"{eval_mode_sel.value}|{_cfg_tag(LR_CONFIGS[0])}"


def _analysis_mode_plot_path(eval_mode_sel: EvalMode) -> str:
    mode_tag = "FT" if eval_mode_sel == EvalMode.FINETUNE else "LP"
    if ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS:
        name = f"comparison_{mode_tag}.png"
    else:
        name = f"_comparison_{mode_tag}_tmp.png"
    return os.path.join(SCRIPT_DIR, name)


def _save_analysis_figure(cache: dict, sections: list[dict], eval_mode_sel: EvalMode) -> None:
    if not ANALYSIS_MODELS:
        guarded_print("[Plot] No models are flagged for analysis; skipping analysis figure.")
        return

    dataset_specs = _ordered_analysis_datasets([section["label"] for section in sections])
    if not dataset_specs:
        guarded_print("[Plot] No datasets available for analysis figure.")
        return

    cache_key = _analysis_cache_key(eval_mode_sel)
    available_entries = []
    for entry in ANALYSIS_MODELS:
        lookup_alias = entry["lookup_alias"]
        found = any(
            cache.get(dataset_label, {}).get("models", {}).get(lookup_alias, {}).get(cache_key) is not None
            for dataset_label, _ in dataset_specs
        )
        if found:
            available_entries.append(entry)
        else:
            guarded_print(
                f"[Plot] Analysis model not found in cache for {eval_mode_sel.value}: {lookup_alias}"
            )

    if not available_entries:
        guarded_print("[Plot] No analysis models were found in the cache; skipping figure.")
        return

    cm_norm_mode = str(ANALYSIS_CM_NORMALIZATION).strip().lower()
    valid_cm_norm_modes = {"none", "true", "pred", "all"}
    if cm_norm_mode not in valid_cm_norm_modes:
        raise ValueError(
            f"Invalid ANALYSIS_CM_NORMALIZATION='{ANALYSIS_CM_NORMALIZATION}'. "
            f"Expected one of {sorted(valid_cm_norm_modes)}."
        )

    def _normalize_cm_for_analysis(cm_total: np.ndarray) -> np.ndarray:
        cm_float = cm_total.astype(np.float32)
        if cm_norm_mode == "none":
            return cm_float
        if cm_norm_mode == "true":
            denom = cm_float.sum(axis=1, keepdims=True)
            return np.divide(
                cm_float,
                denom,
                out=np.zeros_like(cm_float, dtype=np.float32),
                where=denom != 0,
            )
        if cm_norm_mode == "pred":
            denom = cm_float.sum(axis=0, keepdims=True)
            return np.divide(
                cm_float,
                denom,
                out=np.zeros_like(cm_float, dtype=np.float32),
                where=denom != 0,
            )
        if cm_norm_mode == "all":
            total = float(cm_float.sum())
            if total <= 0.0:
                return np.zeros_like(cm_float, dtype=np.float32)
            return cm_float / total
        raise AssertionError(f"Unhandled cm_norm_mode: {cm_norm_mode}")

    def _format_cm_annotation(val: float) -> str:
        if cm_norm_mode == "none":
            return f"{int(round(float(val)))}"
        return f"{float(val):.0%}"

    def _cm_ann_fontsize_from_num_classes(num_classes: int) -> int:
        return 19 if num_classes <= 3 else 16 if num_classes <= 5 else 13

    # Keep CM annotation font size consistent across all datasets/models in a figure.
    # Use the smallest size implied by the most compact matrix (largest class count, e.g. RGZ DR1).
    max_cm_num_classes = 0
    max_cm_cell_count = 0
    for dataset_label, _ in dataset_specs:
        dataset_models = cache.get(dataset_label, {}).get("models", {})
        for entry in available_entries:
            rec = dataset_models.get(entry["lookup_alias"], {}).get(cache_key)
            if rec is None:
                continue
            cm_total_obj = rec.get("cm_total", [])
            cm_arr = np.array(cm_total_obj, dtype=np.int64) if cm_total_obj is not None else np.empty((0, 0), dtype=np.int64)
            cm_n = int(cm_arr.shape[0])
            if cm_n > max_cm_num_classes:
                max_cm_num_classes = cm_n
            if cm_arr.size > 0:
                max_cm_cell_count = max(max_cm_cell_count, int(cm_arr.max()))
    global_cm_ann_fontsize = _cm_ann_fontsize_from_num_classes(max_cm_num_classes or 6)

    def _reorder_for_dataset(dataset_label: str, class_names: list[str], cm_total: np.ndarray):
        pref = ANALYSIS_CLASS_ORDER.get(dataset_label, None)
        if not pref:
            return class_names, cm_total
        idx_by_name = {str(name).strip().lower(): idx for idx, name in enumerate(class_names)}
        seen = set()
        order_idx = []
        for name in pref:
            idx = idx_by_name.get(str(name).strip().lower(), None)
            if idx is not None and idx not in seen:
                order_idx.append(idx)
                seen.add(idx)
        for idx in range(len(class_names)):
            if idx not in seen:
                order_idx.append(idx)
        if len(order_idx) != len(class_names):
            return class_names, cm_total
        ordered_names = [class_names[idx] for idx in order_idx]
        ordered_cm = cm_total[np.ix_(order_idx, order_idx)]
        return ordered_names, ordered_cm

    def _display_class_label(dataset_label: str, label: str) -> str:
        text = str(label).strip()
        if text.lower() in {"relaxed double", "reduce double"}:
            return "Red. Dbl."
        if dataset_label == "RGZ":
            return text.upper()
        return text

    # Compact paper layout using actual table cells (not image blocks),
    # with model blocks separated by double vertical lines.
    if ANALYSIS_COMPACT_TABLE_LAYOUT:
        if cm_norm_mode == "none":
            cm_norm = mpl.colors.Normalize(vmin=0.0, vmax=float(max(max_cm_cell_count, 1)))
        else:
            cm_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cm_cmap = mpl.cm.get_cmap(ANALYSIS_CM_CMAP_NAME)

        dataset_rows = []
        max_table_cols = 0
        gap_cols = 1

        for dataset_label, dataset_display in dataset_specs:
            dataset_cache = cache.get(dataset_label, {})
            dataset_models = dataset_cache.get("models", {})
            cached_class_names = dataset_cache.get("class_names", [])

            ref_record = None
            for entry in available_entries:
                rec = dataset_models.get(entry["lookup_alias"], {}).get(cache_key)
                if rec is not None:
                    ref_record = rec
                    break
            if ref_record is None:
                continue

            ref_cm = np.array(ref_record["cm_total"], dtype=np.int64)
            n_classes = int(ref_cm.shape[0])
            if len(cached_class_names) == n_classes:
                class_names_raw = list(cached_class_names)
            else:
                class_names_raw = [f"C{i + 1}" for i in range(n_classes)]

            ordered_names_raw, _ = _reorder_for_dataset(dataset_label, class_names_raw, ref_cm)
            class_names = [_display_class_label(dataset_label, name) for name in ordered_names_raw]

            cm_blocks = []
            missing_blocks = []
            for entry in available_entries:
                record = dataset_models.get(entry["lookup_alias"], {}).get(cache_key)
                if record is None:
                    cm_blocks.append(None)
                    missing_blocks.append(True)
                    continue

                cm_total = np.array(record["cm_total"], dtype=np.int64)
                names_for_record = list(cached_class_names) if len(cached_class_names) == cm_total.shape[0] else [
                    f"C{i + 1}" for i in range(cm_total.shape[0])
                ]
                _, cm_total = _reorder_for_dataset(dataset_label, names_for_record, cm_total)
                cm_display = _normalize_cm_for_analysis(cm_total)
                cm_blocks.append(cm_display)
                missing_blocks.append(False)

            table_cols = len(available_entries) * n_classes + (len(available_entries) - 1) * gap_cols
            max_table_cols = max(max_table_cols, table_cols)
            dataset_rows.append(
                {
                    "dataset_display": dataset_display,
                    "n_classes": n_classes,
                    "class_names": class_names,
                    "cm_blocks": cm_blocks,
                    "missing_blocks": missing_blocks,
                    "table_cols": table_cols,
                }
            )

        if not dataset_rows:
            guarded_print("[Plot] No dataset rows available for compact analysis table figure.")
            return

        fig_width = max(8.4, 0.38 * max_table_cols + 2.6)
        fig_height = max(6.2, sum(0.42 * row["n_classes"] + 1.0 for row in dataset_rows) + 1.0)
        fig, axes = plt.subplots(
            len(dataset_rows),
            1,
            figsize=(fig_width, fig_height),
            dpi=FIG_DPI,
            gridspec_kw={"height_ratios": [max(2, row["n_classes"]) for row in dataset_rows]},
        )
        axes = np.atleast_1d(axes)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.12, hspace=0.45)

        for row_idx, (ax, row) in enumerate(zip(axes, dataset_rows)):
            n_classes = row["n_classes"]
            table_cols = row["table_cols"]
            class_names = row["class_names"]
            gap_indices = {(i + 1) * n_classes + i for i in range(len(available_entries) - 1)}

            block_starts = []
            col_labels = []
            cell_text = []
            cell_colors = []
            for _ in range(n_classes):
                cell_text.append([])
                cell_colors.append([])

            col_cursor = 0
            for block_idx, cm_display in enumerate(row["cm_blocks"]):
                block_starts.append(col_cursor)
                if cm_display is None:
                    for r in range(n_classes):
                        for c in range(n_classes):
                            cell_text[r].append("N/A" if (r == n_classes // 2 and c == n_classes // 2) else "")
                            cell_colors[r].append((0.92, 0.92, 0.92, 1.0))
                else:
                    for r in range(n_classes):
                        for c in range(n_classes):
                            val = float(cm_display[r, c])
                            cell_text[r].append(_format_cm_annotation(val))
                            cell_colors[r].append(cm_cmap(cm_norm(val)))
                col_labels.extend(class_names)
                col_cursor += n_classes

                if block_idx < len(row["cm_blocks"]) - 1:
                    for r in range(n_classes):
                        cell_text[r].append("")
                        cell_colors[r].append((1.0, 1.0, 1.0, 1.0))
                    col_labels.append("")
                    col_cursor += gap_cols

            ax.axis("off")
            tbl = ax.table(
                cellText=cell_text,
                cellColours=cell_colors,
                rowLabels=class_names,
                colLabels=col_labels,
                cellLoc="center",
                rowLoc="center",
                colLoc="center",
                loc="center",
                bbox=[0.0, 0.0, 1.0, 1.0],
            )

            row_label_w = 0.085
            gap_w = 0.016
            n_data_cols = table_cols - len(gap_indices)
            data_w = max(1e-6, (1.0 - row_label_w - gap_w * len(gap_indices)) / max(n_data_cols, 1))
            col_widths = [gap_w if c in gap_indices else data_w for c in range(table_cols)]

            tbl.auto_set_font_size(False)
            for (r, c), cell in tbl.get_celld().items():
                if c == -1:
                    cell.set_width(row_label_w)
                    cell.set_facecolor("white")
                    cell.set_edgecolor("white")
                    cell.get_text().set_fontsize(9)
                    continue

                cell.set_width(col_widths[c])
                if r == 0:
                    cell.set_facecolor("white")
                    cell.set_edgecolor("white")
                    cell.get_text().set_fontsize(8)
                    cell.get_text().set_rotation(45)
                    cell.get_text().set_ha("right")
                else:
                    if c in gap_indices:
                        cell.set_facecolor("white")
                        cell.set_edgecolor("white")
                        cell.get_text().set_text("")
                    else:
                        cell.set_edgecolor((1.0, 1.0, 1.0, 0.80))
                        cell.set_linewidth(0.35)
                        txt = cell.get_text()
                        txt.set_fontsize(global_cm_ann_fontsize)
                        txt.set_fontweight("bold")
                        raw = txt.get_text().strip()
                        if not raw or raw == "N/A":
                            txt.set_color("black")
                        else:
                            try:
                                if raw.endswith("%"):
                                    val = float(raw[:-1]) / 100.0
                                else:
                                    val = float(raw)
                                r_ch, g_ch, b_ch, _ = cm_cmap(cm_norm(val))
                                luminance = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
                                txt.set_color("black" if luminance > 0.55 else "white")
                            except Exception:
                                txt.set_color("black")

            # Dataset header for this row.
            ax.text(
                -0.015,
                1.03,
                row["dataset_display"],
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

            # Model headers above each block (once, top row only).
            if row_idx == 0:
                x_lefts = [row_label_w]
                for c in range(table_cols):
                    x_lefts.append(x_lefts[-1] + col_widths[c])
                for block_idx, start in enumerate(block_starts):
                    block_w = sum(col_widths[start:start + n_classes])
                    center_x = x_lefts[start] + 0.5 * block_w
                    ax.text(
                        center_x,
                        1.14,
                        available_entries[block_idx]["lookup_alias"],
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

            # Double separators on both sides of the gap columns.
            x_left = row_label_w
            for c in range(table_cols):
                w = col_widths[c]
                if c in gap_indices:
                    ax.plot([x_left, x_left], [0, 1], transform=ax.transAxes, color="black", linewidth=0.9, clip_on=False)
                    ax.plot([x_left + w, x_left + w], [0, 1], transform=ax.transAxes, color="black", linewidth=0.9, clip_on=False)
                x_left += w

        fig.text(0.52, 0.06, "Predicted class", ha="center", va="center", fontsize=10)
        fig.text(0.03, 0.5, "True class", ha="center", va="center", rotation=90, fontsize=10)

        out_file = _analysis_mode_plot_path(eval_mode_sel)
        fig.savefig(out_file, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        if ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS:
            guarded_print(f"Saved comparison plot to {out_file}")
        else:
            guarded_print(f"Saved temporary comparison plot to {out_file}")
        return

    n_cols = len(available_entries)
    n_rows = len(dataset_specs) + max(0, len(dataset_specs) - 1)
    height_ratios = []
    for dataset_idx, _ in enumerate(dataset_specs):
        height_ratios.append(1.20)
        if dataset_idx < len(dataset_specs) - 1:
            height_ratios.append(0.06)

    fig_width = max(10.5, 3.25 * n_cols)
    fig_height = max(8.2, 2.2 * sum(height_ratios))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        dpi=FIG_DPI,
        gridspec_kw={"height_ratios": height_ratios},
    )

    axes = np.array(axes, dtype=object)
    if axes.ndim == 0:
        axes = axes[None, None]
    elif axes.ndim == 1:
        if n_cols == 1:
            axes = axes[:, None]
        else:
            axes = axes[None, :]

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.955,
        bottom=0.09,
        hspace=0.30,
        wspace=0.08,
    )

    if cm_norm_mode == "none":
        cm_norm = mpl.colors.Normalize(vmin=0.0, vmax=float(max(max_cm_cell_count, 1)))
    else:
        cm_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cm_cmap = plt.get_cmap(ANALYSIS_CM_CMAP_NAME)

    row_cursor = 0
    for row_idx, (dataset_label, dataset_display) in enumerate(dataset_specs):
        cm_row = row_cursor
        dataset_cache = cache.get(dataset_label, {})
        dataset_models = dataset_cache.get("models", {})
        cached_class_names = dataset_cache.get("class_names", [])

        for col_idx, entry in enumerate(available_entries):
            lookup_alias = entry["lookup_alias"]
            record = dataset_models.get(lookup_alias, {}).get(cache_key)
            cm_ax = axes[cm_row, col_idx]
            if row_idx == 0:
                # Global model headers: keep them above the first dataset title.
                cm_ax.set_title(lookup_alias, pad=16, fontsize=ANALYSIS_HEADER_FONT_SIZE)

            if record is None:
                cm_ax.axis("off")
                cm_ax.text(
                    0.5,
                    0.5,
                    f"{lookup_alias}\nmissing",
                    ha="center",
                    va="center",
                    fontsize=11,
                    transform=cm_ax.transAxes,
                )
                continue

            cm_total = np.array(record["cm_total"], dtype=np.int64)
            num_classes = int(cm_total.shape[0])
            if len(cached_class_names) == num_classes:
                class_names = list(cached_class_names)
            else:
                class_names = [f"C{i + 1}" for i in range(num_classes)]
            class_names, cm_total = _reorder_for_dataset(dataset_label, class_names, cm_total)
            class_names = [_display_class_label(dataset_label, name) for name in class_names]
            num_classes = int(len(class_names))

            cm_display = _normalize_cm_for_analysis(cm_total)
            cm_ax.imshow(
                cm_display,
                interpolation="nearest",
                cmap=cm_cmap,
                norm=cm_norm,
                aspect="auto",
            )

            cm_ax.set_xticks(np.arange(num_classes))
            cm_ax.set_xticklabels(class_names, rotation=45, ha="right")
            cm_ax.set_yticks(np.arange(num_classes))
            if col_idx == 0:
                cm_ax.set_yticklabels(class_names)
            else:
                cm_ax.set_yticklabels([])
            cm_ax.tick_params(
                axis="both",
                which="major",
                labelsize=8,
                length=0,
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

            # Thin intra-cell grid to improve readability in print.
            cm_ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
            cm_ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
            cm_ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.75)
            cm_ax.tick_params(
                axis="both",
                which="minor",
                length=0,
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

            for (r, c), val in np.ndenumerate(cm_display):
                cm_ax.text(
                    c,
                    r,
                    _format_cm_annotation(float(val)),
                    ha="center",
                    va="center",
                    fontsize=global_cm_ann_fontsize,
                    fontweight="bold",
                    color=(
                        "black"
                        if (0.299 * cm_cmap(cm_norm(float(val)))[0]
                            + 0.587 * cm_cmap(cm_norm(float(val)))[1]
                            + 0.114 * cm_cmap(cm_norm(float(val)))[2]) > 0.55
                        else "white"
                    ),
                )

        left_ax = axes[cm_row, 0]
        right_ax = axes[cm_row, -1]
        # Keep first dataset subtitle clearly below the global model headers.
        if row_idx == 0:
            y_title = left_ax.get_position().y1 + 0.010
        else:
            y_title = left_ax.get_position().y1 + 0.014
        fig.text(
            0.5 * (left_ax.get_position().x0 + right_ax.get_position().x1),
            y_title,
            dataset_display,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        if row_idx < len(dataset_specs) - 1:
            spacer_row = row_cursor + 1
            for col_idx in range(n_cols):
                axes[spacer_row, col_idx].axis("off")
        row_cursor += 1 + (1 if row_idx < len(dataset_specs) - 1 else 0)

    # Keep repeated axis text out of each panel.
    fig.text(0.5, 0.03, "Predicted class", ha="center", va="center", fontsize=10)
    fig.text(0.02, 0.5, "True class", ha="center", va="center", rotation=90, fontsize=10)

    out_file = _analysis_mode_plot_path(eval_mode_sel)
    fig.savefig(out_file, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    if ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS:
        guarded_print(f"Saved comparison plot to {out_file}")
    else:
        guarded_print(f"Saved temporary comparison plot to {out_file}")


def _save_combined_lp_ft_image() -> None:
    lp_path = _analysis_mode_plot_path(EvalMode.LINEAR_PROBE)
    ft_path = _analysis_mode_plot_path(EvalMode.FINETUNE)
    if not (os.path.exists(lp_path) and os.path.exists(ft_path)):
        guarded_print("[Plot] Skipping combined LP+FT image (missing LP or FT PNG).")
        return

    lp = np.asarray(plt.imread(lp_path))
    ft = np.asarray(plt.imread(ft_path))

    def _inner_whitespace_cols(img: np.ndarray, side: str) -> int:
        """Count fully blank/white columns from one side (for tighter LP/FT join)."""
        if img.ndim < 2:
            return 0

        if img.ndim == 2:
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            alpha = None
        else:
            ch = img.shape[2]
            if ch >= 3:
                rgb = img[:, :, :3]
            else:
                rgb = np.repeat(img[:, :, :1], 3, axis=2)
            alpha = img[:, :, 3] if ch >= 4 else None

        h, w = rgb.shape[:2]
        if w <= 1:
            return 0

        white_thr = 0.985
        alpha_empty_thr = 0.02
        indices = range(w) if side == "left" else range(w - 1, -1, -1)
        count = 0
        for c in indices:
            col_rgb = rgb[:, c, :]
            col_white = np.all(col_rgb >= white_thr, axis=1)
            if alpha is not None:
                col_ok = np.logical_or(col_white, alpha[:, c] <= alpha_empty_thr)
            else:
                col_ok = col_white
            if np.all(col_ok):
                count += 1
            else:
                break
        return count

    # Remove only inner margins so separator sits closer to both panels.
    lp_right_blank = _inner_whitespace_cols(lp, "right")
    ft_left_blank = _inner_whitespace_cols(ft, "left")
    keep_inner_pad_cols = 3
    lp_trim = max(0, lp_right_blank - keep_inner_pad_cols)
    ft_trim = max(0, ft_left_blank - keep_inner_pad_cols)
    if lp_trim > 0:
        lp = lp[:, : lp.shape[1] - lp_trim]
    if ft_trim > 0:
        ft = ft[:, ft_trim:]
    lp_h, lp_w = lp.shape[:2]
    ft_h, ft_w = ft.shape[:2]
    max_h = float(max(lp_h, ft_h))

    fig_height = max(6.0, max_h / 320.0)
    fig_width = max(10.0, fig_height * ((lp_w + ft_w) / max_h))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        dpi=FIG_DPI,
        gridspec_kw={"width_ratios": [lp_w, ft_w]},
    )

    axes = np.atleast_1d(axes)
    axes[0].imshow(lp)
    axes[1].imshow(ft)
    axes[0].set_title("(a) Linear Probe", pad=10, fontsize=ANALYSIS_HEADER_FONT_SIZE)
    axes[1].set_title("(b) Fine-Tuning", pad=10, fontsize=ANALYSIS_HEADER_FONT_SIZE)
    for ax in axes:
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.0)
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    x_div = 0.5 * (pos0.x1 + pos1.x0)
    y0 = min(pos0.y0, pos1.y0)
    y1 = max(pos0.y1, pos1.y1)
    divider = plt.Line2D([x_div, x_div], [y0, y1], transform=fig.transFigure, color="black", linewidth=1.0)
    fig.add_artist(divider)

    out_path = os.path.join(SCRIPT_DIR, "comparison_LP_FT.png")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    guarded_print(f"Saved combined comparison plot to {out_path}")

    if not ANALYSIS_SAVE_INDIVIDUAL_MODE_PLOTS:
        for tmp_path in (lp_path, ft_path):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
