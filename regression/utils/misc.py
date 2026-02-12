import os
import importlib.util
import sys
from types import ModuleType

def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename: str) -> ModuleType:
    module_name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"