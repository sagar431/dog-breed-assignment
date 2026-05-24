from __future__ import annotations

import argparse
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from dog_breed_assignment.data import DogBreedDataModule
from dog_breed_assignment.models import DogBreedClassifier
from dog_breed_assignment.utils.runtime import batch_limit, configure_cpu, configure_logging, parse_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dog breed classifier.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--experiment_name", default="dog_breed_classification")
    parser.add_argument("--model_name", default="mobilenetv3_small_050")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=int, default=100, help="0 means full train set.")
    parser.add_argument("--limit_val_batches", type=int, default=50, help="0 means full validation set.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--unfreeze_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu(args.cpu_threads)
    log_path = configure_logging(args.logs_dir)
    logger.info("Training started")
    logger.info("Log file: {}", log_path)
    seed_everything(args.seed, workers=True)

    data_module = DogBreedDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        download=not args.no_download,
    )
    data_module.setup("fit")
    logger.info("Classes: {}", ", ".join(data_module.class_names))

    model = DogBreedClassifier(
        model_name=args.model_name,
        num_classes=data_module.num_classes,
        class_names=data_module.class_names,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="{epoch}-{step}-{val_loss:.4f}",
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        logger=TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment_name),
        callbacks=[RichProgressBar(), RichModelSummary(max_depth=2), checkpoint, LearningRateMonitor("epoch")],
        default_root_dir=Path(args.logs_dir),
        limit_train_batches=batch_limit(args.limit_train_batches),
        limit_val_batches=batch_limit(args.limit_val_batches),
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)
    logger.info("Training finished")
    logger.info("Best checkpoint: {}", checkpoint.best_model_path or checkpoint.last_model_path)


if __name__ == "__main__":
    main()

