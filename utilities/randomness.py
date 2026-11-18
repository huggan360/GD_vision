import random
import torch


def seed_everything(seed):
    """Make random choices repeatable as far as the hardware allows."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
