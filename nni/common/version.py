import logging
try:
    import torch
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
except Exception:
    logging.info("PyTorch is not installed.")
    TORCH_VERSION = None
