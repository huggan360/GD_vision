"""Edit the settings below, then run: python main.py"""

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utilities.preprocessing import PrepareViTImage
from utilities.model import ARCHITECTURE, IMAGE_SIZE, MEAN, STD, SquarePad, head_parameters, initialize_classifier
from utilities.devices import choose_device
from utilities.files import new_run_folder, write_json
from utilities.plots import plot_training_loss
from utilities.randomness import seed_everything
from utilities.data import CompleteDataset, make_loader

ROOT = Path(__file__).resolve().parent


# Settings

DATA = ROOT / "data/complete_dataset"
LABEL_COLUMN = "product_name"
OUTPUT = ROOT / "models/grocery-v2"
INITIALIZE_FROM = None  # Or ROOT / "models/grocery-test/best.pth"
DEVICE = "auto"  # "cuda", "cpu" or "mps", "auto"
SEED = 42
EPOCHS = 15
BATCH_SIZE = 16
WORKERS = 2
LEARNING_RATE = 0.00001
HEAD_LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
FREEZE_BACKBONE_EPOCHS = 2
BALANCED_SAMPLING = False
AMP = True


# Data augmentation

AUGMENTATION = True
CROP_SCALE = (0.7, 1.0)
CROP_RATIO = (0.85, 1.15)
ROTATION_DEGREES = 12
BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.15
ERASING_PROBABILITY = 0.25
ERASING_SCALE = (0.02, 0.15)


def train():
    """Train on every image in the complete dataset, using product names as labels."""
    if EPOCHS < 1:
        raise ValueError("Epochs must be positive.")
    seed_everything(SEED)
    device = choose_device(DEVICE)

    # Load the complete dataset
    train_data = CompleteDataset(DATA, label_column=LABEL_COLUMN)
    print(f"Training on all {len(train_data):,} images with {len(train_data.classes):,} product names.")
    if AUGMENTATION:
        train_data.transform = transforms.Compose(
            [
                PrepareViTImage(),
                SquarePad(),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=CROP_SCALE, ratio=CROP_RATIO, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomRotation(ROTATION_DEGREES, fill=127),
                transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                transforms.RandomErasing(p=ERASING_PROBABILITY, scale=ERASING_SCALE),
            ]
        )
    train_loader = make_loader(train_data, BATCH_SIZE, WORKERS, training=True, balanced=BALANCED_SAMPLING, seed=SEED)

    # Create the output folder

    output = new_run_folder(OUTPUT)
    # Keep the settings and training code beside the model.
    (output / "main.py").write_text(Path(__file__).read_text())
    write_json(output / "classes.json", train_data.classes)

    # Model and optimizer

    model = initialize_classifier(train_data.classes, INITIALIZE_FROM).to(device)
    head = head_parameters(model)
    backbone = [p for p in model.parameters() if all(p is not h for h in head)]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": LEARNING_RATE}, {"params": head, "lr": HEAD_LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1)

    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_loss, history = float("inf"), []

    # Training loop
    for epoch in range(EPOCHS):
        frozen = epoch < FREEZE_BACKBONE_EPOCHS

        for parameter in backbone:
            parameter.requires_grad_(not frozen)

        model.train()

        if frozen:
            for name, module in model.named_children():
                if name != "head":
                    module.eval()

        loss_sum, count, correct = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()

        scheduler.step()

        # Save checkpoints and training progress
        train_loss = loss_sum / count
        accuracy = correct / count
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": accuracy})
        write_json(output / "history.json", history)
        checkpoint = {"format_version": 2, "architecture": ARCHITECTURE, "image_size": IMAGE_SIZE, "model": model.state_dict(), "classes": train_data.classes, "epoch": epoch + 1, "label_column": LABEL_COLUMN, "training_images": len(train_data), "train_loss": train_loss, "train_accuracy": accuracy, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), }
        torch.save(checkpoint, output / "last.pth")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(checkpoint, output / "best.pth")

        plot_training_loss(history, output / "training_loss.png")
        print(f"Epoch {epoch + 1}/{EPOCHS}: train loss={train_loss:.4f}, "f"train accuracy={accuracy:.3f}")

    return output / "best.pth"


# Run
if __name__ == "__main__":
    print(train())
