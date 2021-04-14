import torch


def get_device():
    """Returns the appropriate device depending on what's available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
