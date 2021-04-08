import torch
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using CUDA!")
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy
    
def torch_load(load_path):
    if use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location = 
                          lambda storage, location: storage)
    
def load_state_dict_incomplete(model, state_dict, child = False):
    if child:
        state_dict = {k[k.index('.')+1:]: v for k, v in state_dict.items() if '.' in k}

    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
def partial_state_dict(state_dict): # stores mask and head parameters
    partial_state_dict = OrderedDict()
    for name in state_dict.keys():
        if 'mask_scores' in name or 'bert' not in name:
            partial_state_dict[name] = state_dict[name]
    return partial_state_dict