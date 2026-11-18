"""Edit the settings below, then run: python evaluate.py"""

from pathlib import Path

from utilities.model import load_classifier
from utilities.data import CompleteDataset, make_loader
from utilities.metrics import evaluate_model
from utilities.devices import choose_device
from utilities.files import write_json

ROOT = Path(__file__).resolve().parent

# Settings
DATA = ROOT / "data/complete_dataset"
LABEL_COLUMN = "product_name"
CHECKPOINT = ROOT / "models/grocery-test/best.pth"
OUTPUT = ROOT / "models/grocery-test/evaluation_training.json"
SPLIT = None  # All images. They were used for training, so this measures training fit.
DEVICE = "auto"
BATCH_SIZE = 16
WORKERS = 2


# Load data and evaluate
def evaluate():
    device = choose_device(DEVICE)
    model, classes = load_classifier(CHECKPOINT, device)
    dataset = CompleteDataset(DATA, label_column=LABEL_COLUMN, classes=classes, split=SPLIT)
    loader = make_loader(dataset, BATCH_SIZE, WORKERS)
    metrics = evaluate_model(model, loader, device, classes)
    write_json(OUTPUT, metrics)
    print(f"Dataset accuracy: {metrics['accuracy']:.3f}; macro F1: {metrics['macro_f1']:.3f}")
    return metrics


# Run
if __name__ == "__main__":
    evaluate()
