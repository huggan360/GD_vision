"""Run from this repo: python compile/export.py. Copy the resulting vit.pth into GD_software/vit/."""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.model import load_classifier

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/grocery-test/best.pth"
OUTPUT = CHECKPOINT.parent / "vit.pth"


class CropPreparation(nn.Module):
    """Keep RGB preparation inside the model so C++ only hands over a crop."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        rgb = image.to(torch.float32) / 255.0
        highest, lowest = rgb.amax(2, keepdim=True), rgb.amin(2, keepdim=True)
        saturation = (highest - lowest) / highest.clamp(min=1e-6)
        grey = rgb[:, :, 0:1] * 0.299 + rgb[:, :, 1:2] * 0.587 + rgb[:, :, 2:3] * 0.114
        vibrant = grey + (rgb - grey) * (1.0 + 0.08 * (1.0 - saturation))
        rgb = (vibrant.clamp(0, 1) * 255).round()
        # Pillow's contrast enhancement uses a rounded mean of its greyscale image.
        luminance = ((rgb[:, :, 0] * 19595 + rgb[:, :, 1] * 38470 + rgb[:, :, 2] * 7471 + 32768) / 65536).floor()
        average = (luminance.mean() + 0.5).floor()
        rgb = (average + (rgb - average) * 1.05).clamp(0, 255).floor()
        height, width = rgb.shape[0], rgb.shape[1]
        size = max(height, width)
        left, top = (size - width) // 2, (size - height) // 2
        tensor = rgb.permute(2, 0, 1).unsqueeze(0)
        tensor = F.pad(tensor, [left, size - width - left, top, size - height - top], value=127.0)
        tensor = F.interpolate(tensor, size=[224, 224], mode="bilinear", align_corners=False, antialias=True).round()
        return (tensor / 255.0 - 0.5) / 0.5


class RuntimeClassifier(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.prepare = CropPreparation()
        self.classifier = classifier

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.prepare(rgb))


def export(checkpoint=CHECKPOINT, output=OUTPUT):
    torch.set_num_threads(4)
    model, classes = load_classifier(checkpoint, torch.device("cpu"))
    # Tracing the fixed ViT input avoids scripting timm's training-only code.
    model = torch.jit.trace(model, torch.zeros(1, 3, 224, 224), check_trace=True)
    runtime = torch.jit.script(RuntimeClassifier(model).eval())
    metadata = {"format": "gd-vit-rgb-v1", "classes": classes, "input": "uint8 RGB HWC", "output": "logits"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(runtime, str(output), _extra_files={"metadata.json": json.dumps(metadata)})
    # Verify loading and both landscape and portrait crops before handing it over.
    loaded = torch.jit.load(str(output), map_location="cpu")
    with torch.inference_mode():
        for shape in ((120, 80, 3), (64, 180, 3)):
            sample = torch.randint(0, 256, shape, dtype=torch.uint8)
            torch.testing.assert_close(runtime(sample), loaded(sample))
    print(output)
    return output


if __name__ == "__main__":
    export()
