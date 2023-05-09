import os
import random
import numpy as np
import torch

# call before train
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# dataset config
DATA_DIR = './data/'
TRAIN_ROOT_PATH = DATA_DIR + 'train'

# train cache
CACHE_DIR = './cache/'

