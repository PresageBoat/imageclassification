import os
import torch
from typing import List



def select_device(device:List[int]):
    cpu=device[0]=='-1'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
    elif device:
        devices=','.join(map(str,device))
        os.environ['CUDA_VISIBLE_DEVICES']=devices
        err_str='CUDA unavailable, invalid device {devices} requested'
        assert torch.cuda.is_available() ,err_str

    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == '__main__':
    select_device('0,2,3',48)
