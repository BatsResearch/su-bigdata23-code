import logging
import torch
import numpy as np
import torch




def init_random(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info('random seed: %d', seed)

def remove_LF(ds, lf_num_list):
    for i in ds.weak_labels:
        for index in sorted(lf_num_list, reverse=True):
            del i[index]
    return ds