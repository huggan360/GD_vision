# READ CLASSIFICATION DATA WITH A FIXED CATEGORY ORDER
from collections import Counter
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utilities.model import image_transform

class CompleteDataset(Dataset):
    """Read images and product names directly from complete_dataset/annotations.csv."""

    def __init__(self, root, label_column="product_name", classes=None, split=None):
        self.root = Path(root)
        with (self.root / "annotations.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image", label_column} | ({"split"} if split else set())
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"annotations.csv needs columns: {sorted(required)}")
            rows = [row for row in reader if split is None or row["split"] == split]
        if not rows:
            raise ValueError("No images found in annotations.csv for this split.")
        names = [row[label_column].strip() for row in rows]
        if not all(names):
            raise ValueError(f"Every image needs a nonempty {label_column}.")
        self.classes = list(classes) if classes is not None else sorted(set(names))
        indices = {name: index for index, name in enumerate(self.classes)}
        unknown = set(names) - set(indices)
        if unknown:
            raise ValueError(f"Labels absent from the trained model: {sorted(unknown)[:5]}")
        self.samples = [(self.root / row["image"], indices[name]) for row, name in zip(rows, names)]
        for path, _ in self.samples:
            if not path.is_file():
                raise ValueError(f"Missing product image: {path}")
        self.transform = image_transform()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


# Batch loading
def make_loader(dataset, batch_size, workers, training=False, balanced=False, seed=42):
    """Build batches, optionally showing rare categories more often."""
    if batch_size < 1 or workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative.")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced:
        counts = Counter(label for _, label in dataset.samples)
        weights = [1.0 / counts[label] for _, label in dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights), generator=generator)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=training and sampler is None, sampler=sampler, generator=generator, pin_memory=torch.cuda.is_available(),)
