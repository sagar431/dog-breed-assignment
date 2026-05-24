from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


KAGGLE_DATASET = "khushikhushikhushi/dog-breed-image-dataset"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DogBreedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 8,
        image_size: int = 160,
        num_workers: int = 0,
        val_split: float = 0.2,
        seed: int = 42,
        download: bool = True,
        dataset: str = KAGGLE_DATASET,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.download = download
        self.dataset = dataset
        self.class_names: list[str] = []
        self.num_classes = 0
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        if self._find_split_root(("train", "training")) and self._find_split_root(("val", "valid", "validation", "test")):
            return
        if self._find_imagefolder_root() is not None:
            return
        if not self.download:
            return

        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        archive_path = raw_dir / "dog-breed-image-dataset.zip"

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset, path=str(raw_dir), quiet=False)

        downloaded_archives = sorted(raw_dir.glob("*.zip"))
        if downloaded_archives:
            archive_path = downloaded_archives[0]
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(raw_dir)

    def setup(self, stage: str | None = None) -> None:
        self.prepare_data()
        train_transform = self.train_transform(self.image_size)
        eval_transform = self.eval_transform(self.image_size)

        train_root = self._find_split_root(("train", "training"))
        val_root = self._find_split_root(("val", "valid", "validation", "test"))

        if train_root is not None and val_root is not None:
            self.train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
            self.val_dataset = datasets.ImageFolder(val_root, transform=eval_transform)
            self.class_names = self.train_dataset.classes
        else:
            root = self._find_imagefolder_root()
            if root is None:
                raise FileNotFoundError(
                    f"No dog-breed class folders found under {self.data_dir}. "
                    "Set KAGGLE_USERNAME/KAGGLE_KEY or mount a prepared ImageFolder dataset."
                )
            base_dataset = datasets.ImageFolder(root)
            train_indices, val_indices = self._split_indices(len(base_dataset))
            self.train_dataset = Subset(datasets.ImageFolder(root, transform=train_transform), train_indices)
            self.val_dataset = Subset(datasets.ImageFolder(root, transform=eval_transform), val_indices)
            self.class_names = base_dataset.classes

        self.num_classes = len(self.class_names)
        if self.num_classes < 2:
            raise ValueError("Dog breed training needs at least two breed folders.")

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def _dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def _find_split_root(self, names: Sequence[str]) -> Path | None:
        for candidate in self.data_dir.rglob("*"):
            if candidate.is_dir() and candidate.name.lower() in names and self._is_imagefolder_root(candidate):
                return candidate
        return None

    def _find_imagefolder_root(self) -> Path | None:
        if self._is_imagefolder_root(self.data_dir):
            return self.data_dir

        candidates = [
            path
            for path in self.data_dir.rglob("*")
            if path.is_dir() and self._is_imagefolder_root(path)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: len(self._class_dirs(path)))

    @staticmethod
    def _is_imagefolder_root(path: Path) -> bool:
        return len(DogBreedDataModule._class_dirs(path)) >= 2

    @staticmethod
    def _class_dirs(path: Path) -> list[Path]:
        if not path.exists():
            return []
        return [
            child
            for child in path.iterdir()
            if child.is_dir()
            and child.name.lower() not in SPLIT_NAMES
            and any(
                image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
                for image_path in child.rglob("*")
            )
        ]

    def _split_indices(self, dataset_size: int) -> tuple[list[int], list[int]]:
        if dataset_size < 2:
            raise ValueError("Need at least two images to create a validation split.")
        val_size = max(1, int(dataset_size * self.val_split))
        train_size = dataset_size - val_size
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        return indices[:train_size], indices[train_size:]

    @staticmethod
    def train_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @staticmethod
    def eval_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
