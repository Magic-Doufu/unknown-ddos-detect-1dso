m_lst = [None, 'M1', 'M2']
ver_lst = { 'M1': 'ISCX',
            'M2': 'REVI'}
import os, matplotlib.pyplot as plt
import numpy as np
def fixed_seed(seed):
    print(f'固定隨機種子:{seed}')
    import torch, random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def pathCreate(path):
    print(f'檢查路徑是否存在{path}')
    os.makedirs(path, exist_ok=True)