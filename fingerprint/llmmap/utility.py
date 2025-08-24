import pickle
import hashlib
import os, random
import re
import torch

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        ...

def _hash(input_string):
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    integer_hash = int(sha256_hash, 16)
    return integer_hash

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(path, data):
    with open(path, 'wb') as f:
        data = pickle.dump(data, f)

def sample_from_multi_universe(universe):
    sample = {}
    for k, u in universe.items():
        sample[k] = random.sample(u, 1)[0]
    return sample


def set_gpus(gpus, with_pytorch_gpu_optimization=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    if with_pytorch_gpu_optimization:
        pytorch_gpu_optimization()

def pytorch_gpu_optimization():
    if torch.cuda.is_available():
        try:
            # Set memory allocation strategy for better memory management
            torch.cuda.empty_cache()
            # Get number of available GPUs
            num_gpus = torch.cuda.device_count()
            # ...existing code...
        except RuntimeError:
            pass
