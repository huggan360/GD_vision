from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utilities.preprocessing import PrepareViTImage

ARCHITECTURE = "vit_small_patch16_224"
PRETRAINED = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMAGE_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


# Crop preprocessing
class SquarePad:
    """Keep the whole product by adding a grey border before resizing."""

    def __call__(self, image):
        width, height = image.size
        size = max(width, height)
        left, top = (size - width) // 2, (size - height) // 2
        return transforms.functional.pad(image, (left, top, size - width - left, size - height - top), fill=127)


def image_transform():
    """Apply the fixed crop preparation used for evaluation."""
    return transforms.Compose([PrepareViTImage(), SquarePad(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# Build the classifier
def build_classifier(number_of_classes, pretrained=True, cache_dir=None):
    """Keep the pretrained ViT body and replace its final category layer."""
    import timm

    if number_of_classes < 2:
        raise ValueError("Provide at least two product categories.")
    cache_dir = cache_dir or Path(__file__).resolve().parents[1] / "models/pretrained"
    return timm.create_model(PRETRAINED, pretrained=pretrained, num_classes=number_of_classes, cache_dir=cache_dir)


def head_parameters(model):
    """The new category layer; everything else is the pretrained body."""
    return list(model.get_classifier().parameters())


# SAVE, LOAD, AND REUSE TRAINED CATEGORIES
def read_checkpoint(path):
    """Read our classifier format without unpickling arbitrary Python objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("format_version") != 2:
        raise ValueError("Expected a ViT-Small/16 training checkpoint, format version 2.")
    classes = checkpoint["classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Checkpoint has invalid category names.")
    return checkpoint


def load_classifier(path, device):
    """Load a trained model with the exact label order used in training."""
    checkpoint = read_checkpoint(path)
    model = build_classifier(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["classes"]


# Initialize from an earlier grocery checkpoint
def initialize_classifier(classes, previous_path=None):
    """Start from ImageNet or reuse an earlier grocery model for fine-tuning."""
    if not previous_path:
        return build_classifier(len(classes), pretrained=True)
    checkpoint = read_checkpoint(previous_path)
    model = build_classifier(len(classes), pretrained=False)
    state = checkpoint["model"].copy()
    old_weight = state.pop("head.weight")
    old_bias = state.pop("head.bias")
    model.load_state_dict(state, strict=False)
    old_indices = {name: i for i, name in enumerate(checkpoint["classes"])}
    with torch.no_grad():
        for index, name in enumerate(classes):
            if name in old_indices:
                model.head.weight[index].copy_(old_weight[old_indices[name]])
                model.head.bias[index].copy_(old_bias[old_indices[name]])
    return model
