import os
import warnings
import itertools
import torch

import torch.distributed as dist

from collections import deque
from copy import deepcopy


def is_main_process():
    if dist.is_available():
        return dist.get_rank() == 0
    return True 


def load_deque(d1, d2):
    """Load d2 into d1
    """
    assert type(d1) == type(d2)
    if type(d1) is Deque:
        load_deque(d1.deque, d2.deque)
        d1.total = d2.total
        return
    m = d1.maxlen
    M = len(d2)
    if m <= M:
        d1.clear()
        d1.extend(itertools.islice(d2, M-m, M))
    else:
        d1.clear()
        d1.extend(d2)

def save_safely(obj, path):
    if not is_main_process():
        return
    # Write first to a temporary file
    temp_path = str(path) + '.tmp'
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    torch.save(obj, temp_path)
    # Safely rename to the main file
    os.rename(temp_path, path)


class Deque(object):
    """
    Custom deque class that can save store its elements on disk instead of in memory
    Args:
        path: directory to save the elements
    """
    def __init__(self, maxlen=None, dir=None, keep_all=False) -> None:
        super().__init__()
        self.dir = dir
        if self.dir is not None:
            os.makedirs(self.dir, exist_ok=True)
            
        self.deque = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.keep_all = keep_all
        self.total = 0

    def append(self, x: torch.Tensor):
        if self.dir is None:
            # add to the tensor deque and keep in memory
            self.deque.append(x)
        else:
            # save to disk and keep only its path in memory
            if self.keep_all:
                # if we keep all the checkoints, then it's simple
                name = f'weight{self.total}.pt'
                save_safely(x, os.path.join(self.dir, name))
                self.deque.append(name)
            else:
                # if we want to keep only N (maxlen) checkpoints on disk, then
                # the files are named weight0.pt, weight1.pt,..., weight{N-1}.pt, 
                if len(self.deque) >= self.deque.maxlen:
                    # if the deque is already full, we insert the new element into it
                    # using the name of the head (which will be popped out)
                    save_safely(x, os.path.join(self.dir, self.deque[0]))
                    self.deque.rotate(-1)
                else:
                    name = f'weight{self.total % self.maxlen}.pt'
                    assert name not in self.deque
                    save_safely(x, os.path.join(self.dir, name))
                    self.deque.append(name)
        
        self.total += 1

    def clear(self):
        self.deque.clear()
        self.total = 0

    def __getitem__(self, key):
        if self.dir is None:
            return self.deque[key]
        else:
            return torch.load(os.path.join(self.dir, self.deque[key]), map_location='cpu')

    def __len__(self):
        return len(self.deque)
    

class AveragedModel(torch.nn.Module):
    """
    Model Averaging, including Exponential Moving Average and Uniform Average
    Args:
        method (str, optional): 'avg' for uniform average and 'ema' for exponential moving average
    """
    def __init__(self, model, method='avg', decay=0.999, last_n=-1, last_n_dir=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        model = model.init_model_avg()
        self.module = model
        for p in self.module.parameters():
            p.requires_grad = False
            p.detach_()
        self.module.eval()

        self.avg_tensors = list(self.module.state_dict().values())
        for x in self.avg_tensors:
            x.detach_()

        self.model_tensors = list(model.state_dict().values())
        for x in self.model_tensors:
            x.detach_()

        self.last_n = last_n

        # (last_n - 1) because the last iterate is the current model
        # it doesn't have to be part of the deque for calculations
        self.tensor_deque = Deque(maxlen=(last_n-1), dir=last_n_dir) if self.last_n > 2 else None
        self.n_averaged = 0

        # self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long))

        # Update function for EMA
        @torch.no_grad()
        def ema_update(avg_tensors, model_tensors):
            # The combination of these two
            # torch._foreach_mul_(avg_tensors, scalar=decay)
            # torch._foreach_add_(avg_tensors, model_tensors, alpha=(1. - decay))
            # is equivalent to
            torch._foreach_lerp_(avg_tensors, model_tensors, weight=(1. - decay))
        
        # Update function for uniform average, without last-n
        @torch.no_grad()
        def avg_update(avg_tensors, model_tensors):
            torch._foreach_lerp_(avg_tensors, model_tensors, weight=1.0/(1.0 + self.n_averaged))

        # Update function for uniform average, wit last-n
        # Denote 
        #     N = last_n (which is maxlen of the deque). Here N > 0.
        #     t = current epoch index
        #     n = current length of the deque
        # Then:
        #     If n < N: p_avg <- p_avg + (p_t - p_avg)/(n+1)
        #     If n = N: p_avg <- p_avg + (p_t - p_{t-n})/N,
        #     where p_{t-n} is the head of the deque.
        @torch.no_grad()
        def avg_update_last_n(avg_tensors, model_tensors):
            if self.n_averaged < last_n:
                torch._foreach_lerp_(avg_tensors, model_tensors, weight=1.0/(1.0 + self.n_averaged))
            else:
                # get the head of the deque and move it to the correct device
                head_deque_tensors = [x.to(avg_tensors[0].device) for x in self.tensor_deque[0]]
                diffs = torch._foreach_sub(model_tensors, head_deque_tensors)
                torch._foreach_add_(avg_tensors, diffs, alpha=1./last_n)
            
            # Add the current parameters to the deque
            # always store the deque on CPU to save GPU memory
            self.tensor_deque.append([x.cpu().clone() for x in model_tensors])

        if method == 'ema':
            assert decay > 0
            self.update_fn = ema_update
        elif method == 'avg':
            if last_n < 1:
                self.update_fn = avg_update
            else:
                self.update_fn = avg_update_last_n

    @torch.no_grad()
    def reset(self):
        """Reset to the current state of model
        """
        for ema_v, model_v in zip(self.avg_tensors, self.model_tensors):
            ema_v.copy_(model_v)
        self.n_averaged = 0
        if self.tensor_deque is not None:
            self.tensor_deque.clear()

    @torch.no_grad()
    def update(self):
        self.update_fn(self.avg_tensors, self.model_tensors)
        self.n_averaged += 1

    def state_dict(self, *args, **kwargs):
        return {
            'module': self.module.state_dict(),
            'n_averaged': self.n_averaged,
            'tensor_deque': self.tensor_deque
        }
        
    def load_state_dict(self, state_dict, strict: bool = True):
        self.module.load_state_dict(state_dict['module'], strict)
        self.n_averaged = state_dict['n_averaged']
        print(f'n_averaged so far: {self.n_averaged}')
        # Load checkpoint deque
        # Instead of doing self.tensor_deque = state_dict['tensor_deque']
        # We will load manually, which allows changing the number of checkpoints
        # For example, after doing model averaging over 20 checkpoints during the first few epochs,
        # one may decide to change it to only 10 (due to memory issue for example)
        if self.tensor_deque is not None:
            if self.tensor_deque.maxlen != len(state_dict['tensor_deque']):
                warnings.warn(f"Model averaging: length of saved deque ({len(state_dict['tensor_deque'])}) "
                              f"is different from maxlen of current deque ({self.tensor_deque.maxlen}).")
            load_deque(self.tensor_deque, state_dict['tensor_deque'])
            print(f'length of tensor_deque so far: {len(self.tensor_deque)}')
            if isinstance(self.tensor_deque, Deque) and self.tensor_deque.dir is not None:
                print(f'The tensor deque is stored on disk. All the current files:\n{self.tensor_deque.deque}')

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    