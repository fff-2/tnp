import os
import importlib.util
import math
import torch
from typing import Callable, Any

def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename: str) -> Any:
    module_name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "models/cnp.py">


def logmeanexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x: torch.Tensor, num_samples: int = None, dim: int = 0) -> torch.Tensor:
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration: float) -> str:
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{int(hours)}hrs {int(mins)}mins {int(secs)}secs"


def one_hot(x: torch.Tensor, num: int) -> torch.Tensor:  # [B,N] -> [B,N,num]
    B, N = x.shape
    _x = torch.zeros([B, N, num], dtype=torch.float32, device=x.device)
    for b in range(B):
        for n in range(N):
            i = x[b, n].long()
            _x[b, n, i] = 1.0
    return _x