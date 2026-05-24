from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.table import Table

from dog_breed_assignment.data import DogBreedDataModule
from dog_breed_assignment.models import DogBreedClassifier
from dog_breed_assignment.utils.runtime import configure_cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a dog breed checkpoint on validation data.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--limit_batches", type=int, default=0, help="0 means full validation set.")
    parser.add_argument("--no_download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_module = DogBreedDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    data_module.setup("validate")

    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path, pretrained=False, map_location=device)
    model.eval().to(device)

    total = 0
    correct = 0
    total_loss = 0.0
    batches = 0

    for images, labels in data_module.val_dataloader():
        images = images.to(device)
        labels = labels.to(device)
        with torch.inference_mode():
            logits = model(images)
            loss = model.criterion(logits, labels)
            predictions = logits.argmax(dim=1)
        batch_size = labels.numel()
        total += batch_size
        correct += (predictions == labels).sum().item()
        total_loss += loss.item() * batch_size
        batches += 1
        if args.limit_batches > 0 and batches >= args.limit_batches:
            break

    accuracy = correct / total if total else 0.0
    mean_loss = total_loss / total if total else 0.0

    table = Table(title="Validation Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("val_loss", f"{mean_loss:.6f}")
    table.add_row("val_acc", f"{accuracy:.6f}")
    table.add_row("correct", str(correct))
    table.add_row("total", str(total))
    Console().print(table)


if __name__ == "__main__":
    main()

