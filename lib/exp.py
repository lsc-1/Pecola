import numpy as np

def seed_everything(seed):
    import random, os
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

NUM_LABELS_MAP = {"hc3": 2, "gpt3.5":2, "gpt2": 2,"grover":2}

def get_num_labels(dataset):
    if 'clinc' in dataset:
        return 150
    dataset = dataset.replace('_fewshot','')
    return NUM_LABELS_MAP[dataset]

def take_mean_and_reshape(x): return np.array(
    x).mean(axis=0).reshape(-1)
