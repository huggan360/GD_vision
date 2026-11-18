import torch


def choose_device(name="auto"):
    """Use a GPU when available, otherwise use the CPU."""
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot use it.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but PyTorch cannot use it.")
    return device
