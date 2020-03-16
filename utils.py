import torch
import random
import numpy as np


def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if not opt.cpu:
        torch.cuda.manual_seed_all(opt.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()