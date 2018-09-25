import torch
import random
import numpy as np
from .dataset import YearBookDataset

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_class_counts(opts, split='train'):
    """
    Gets the class counts for each class
    """
    dataset = YearBookDataset(opts.data_dir, split=split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    map_cnt = {l:0 for l in range(opts.nclasses)}

    for d in dataloader:
        map_cnt[d['label'].data.numpy().item(0)] += 1

    return map_cnt.values()


