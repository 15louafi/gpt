import torch

# Reproducibility
torch.manual_seed(42)

# Data & paths
INPUT_PATH: str = "input.txt"

# Training hyperparameters
BATCH_SIZE: int = 64
CONTEXT_LENGTH: int = 256
LEARNING_RATE: float = 3e-4
EVAL_ITERS: int = 100
TRAIN_ITERS: int = 10000

# Model hyperparameters
N_EMBED: int = 284
DROPOUT: float = 0.2
N_HEAD: int = 6
N_LAYER: int = 6

# Device
device: str = "cuda" if torch.cuda.is_available() else "cpu"
