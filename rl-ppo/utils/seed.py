# utils/seed.py
import os, random, numpy as np, torch

def set_seed(seed: int, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Make hash seed deterministic, too
    os.environ["PYTHONHASHSEED"] = str(seed)
