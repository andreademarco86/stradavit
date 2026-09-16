import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.strada_datasets import LoTSSDR2HortonDataset, MiraBestDatasetV2, RadioGalaxyDataset
from utils.radioastroprocessor import RadioFoundationProcessor


# Knobs
EXAMPLES_PER_CLASS = 8
IMAGE_SIZE = 224
RANDOM_SEED = 25
OUTPUT_PATH = PROJECT_ROOT / "utils" / "Test_Views.png"
RADIO_PROCESSOR = RadioFoundationProcessor(image_size=IMAGE_SIZE)
DATASET_FONT_SIZE = 36
CLASS_FONT_SIZE = 32
SHOW_DATASET_TITLES = False


DATASETS = [
    (
        "RGZ",
        RadioGalaxyDataset,
        PROJECT_ROOT / "classification" / "Data" / "rgz_dr1_data" / "cutouts_pad1.50_minpix8",
    ),
    (
        "MiraBest",
        MiraBestDatasetV2,
        PROJECT_ROOT / "classification" / "Data" / "mirabestv2",
    ),
    (
        "LoTSS-Horton",
        LoTSSDR2HortonDataset,
        PROJECT_ROOT / "classification" / "Data" / "lotss_dr2_horton_hires_cutouts",
    ),
]


def load_font(size: int):
    # Prefer STIX-like fonts for a LaTeX-ish look, fall back to common serif fonts.
    candidates = [
        "STIXGeneral.ttf",
        "STIXGeneralRegular.ttf",
        "STIX Two Text.ttf",
        "STIXTwoText-Regular.ttf",
        "Times New Roman.ttf",
        "DejaVuSerif.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(font, text: str):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def collect_indices_by_class(dataset):
    if hasattr(dataset, "samples"):
        labels = [label for _, label in dataset.samples]
    elif hasattr(dataset, "targets"):
        labels = list(dataset.targets)
    else:
        labels = [int(dataset[idx]["labels"]) for idx in range(len(dataset))]

    by_class = {class_idx: [] for class_idx in range(len(dataset.classes))}
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)
    return by_class


def sample_examples(dataset):
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices_by_class = collect_indices_by_class(dataset)
    sampled = {}

    for class_idx, indices in indices_by_class.items():
        if len(indices) <= EXAMPLES_PER_CLASS:
            sampled[class_idx] = indices
            continue
        order = torch.randperm(len(indices), generator=generator).tolist()
        sampled[class_idx] = [indices[i] for i in order[:EXAMPLES_PER_CLASS]]

    return sampled


def reshape_mirabest_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2 and image.shape[0] == 1:
        image = image.reshape(-1)
    if image.ndim == 1:
        side = int(round(image.size ** 0.5))
        if side * side != image.size:
            raise ValueError(f"Cannot reshape MiraBest sample of size {image.size} into a square image.")
        image = image.reshape(side, side)
    return image


def processor_tensor_to_tile(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor[0]
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def get_display_image(dataset_name, dataset, sample_idx):
    if dataset_name == "MiraBest":
        raw_image = reshape_mirabest_image(dataset.data[sample_idx])
        tensor = RADIO_PROCESSOR.process_numpy(raw_image, resize=True)
    else:
        fits_path, _ = dataset.samples[sample_idx]
        tensor = RADIO_PROCESSOR(fits_path)

    return processor_tensor_to_tile(tensor)


def build_panels():
    panels = []
    for dataset_name, dataset_cls, dataset_root in DATASETS:
        dataset = dataset_cls(root=str(dataset_root), transform=None, image_size=IMAGE_SIZE)
        sampled = sample_examples(dataset)

        rows = []
        for class_idx, class_name in enumerate(dataset.classes):
            images = []
            for sample_idx in sampled[class_idx]:
                images.append(get_display_image(dataset_name, dataset, sample_idx))
            rows.append({"class_name": str(class_name).upper(), "images": images})

        panels.append({"dataset_name": dataset_name, "rows": rows})
    return panels


def render_panels(panels, show_dataset_titles: bool = True):
    cols = max(max((len(row["images"]) for row in panel["rows"]), default=0) for panel in panels)
    cols = max(cols, 1)

    margin = 24
    col_gap = 10
    row_gap = 10
    panel_gap = 28

    dataset_font = load_font(DATASET_FONT_SIZE)
    class_font = load_font(CLASS_FONT_SIZE)

    dataset_titles = []
    if show_dataset_titles:
        for panel_idx, panel in enumerate(panels):
            letter = chr(ord("a") + panel_idx)
            dataset_titles.append(f"({letter}) {panel['dataset_name']}")
    else:
        dataset_titles = [""] * len(panels)

    dataset_title_h = max(text_size(dataset_font, title)[1] for title in dataset_titles if title) + 16 if show_dataset_titles else 0

    label_width = max(
        text_size(class_font, row["class_name"])[0]
        for panel in panels
        for row in panel["rows"]
    ) + 24
    grid_width = cols * IMAGE_SIZE + (cols - 1) * col_gap
    canvas_width = margin * 2 + label_width + grid_width
    canvas_height = margin * 2

    for panel in panels:
        canvas_height += dataset_title_h
        canvas_height += len(panel["rows"]) * IMAGE_SIZE
        canvas_height += max(0, len(panel["rows"]) - 1) * row_gap
        canvas_height += panel_gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    y = margin
    for panel, title in zip(panels, dataset_titles):
        if show_dataset_titles:
            title_w, title_h = text_size(dataset_font, title)
            title_x = max(margin, (canvas_width - title_w) // 2)
            draw.text((title_x, y + (dataset_title_h - title_h) // 2), title, fill="black", font=dataset_font)
            y += dataset_title_h

        for row in panel["rows"]:
            class_y = y + (IMAGE_SIZE - text_size(class_font, row["class_name"])[1]) // 2
            draw.text((margin, class_y), row["class_name"], fill="black", font=class_font)

            x = margin + label_width
            for col_idx in range(cols):
                box = (x, y, x + IMAGE_SIZE, y + IMAGE_SIZE)
                draw.rectangle(box, outline="#cccccc", width=1)
                if col_idx < len(row["images"]):
                    tile = row["images"][col_idx]
                    canvas.paste(tile, box[:2])
                x += IMAGE_SIZE + col_gap

            y += IMAGE_SIZE + row_gap

        y += panel_gap - row_gap

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)


def main():
    panels = build_panels()
    render_panels(panels, show_dataset_titles=SHOW_DATASET_TITLES)
    print(f"Saved preview grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
