from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dog_breed_assignment.data import DogBreedDataModule
from dog_breed_assignment.models import DogBreedClassifier
from dog_breed_assignment.utils.images import iter_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dog breed inference on images.")
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def save_prediction(image: Image.Image, output_path: Path, label: str, confidence: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{label} ({confidence:.2%})", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path, pretrained=False, map_location=device)
    model.eval().to(device)
    transform = DogBreedDataModule.eval_transform(args.image_size)

    image_paths = iter_image_paths(args.input_folder, limit=args.limit)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor)[0], dim=0)
            confidence, index = torch.max(probabilities, dim=0)
        label = model.class_names[index.item()]
        save_prediction(image, output_folder / f"{image_path.stem}_prediction.png", label, confidence.item())

    print(f"Saved {len(image_paths)} prediction image(s) to {output_folder}")


if __name__ == "__main__":
    main()
