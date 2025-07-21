import os
import yaml
import argparse
import random
import logging
import numpy as np
import datetime
import torch
from accelerate import Accelerator
from transformers import set_seed


def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def load_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu", "0,1,2")')
    parser.add_argument('--benchmark_config', type=str, default='config/benchmark_config.yaml', help='Path to the benchmark configuration file')
    parser.add_argument('--fingerprint_config', type=str, default='config/llmmap.yaml', help='Path to the fingerprint configuration file')
    parser.add_argument('--log_path', type=str, default='logs/', help='Path to save logs and results')
    parser.add_argument('--fingerprint_method', type=str, default='llmmap', help='Fingerprinting method to use (e.g., "llmmap")')

    args = parser.parse_args()
    return args


def setup_environment(args):
    """Setup random seed and device"""
    # Set random seed
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device configuration for accelerator
    if args.device == 'cpu':
        # Force CPU usage by setting CUDA_VISIBLE_DEVICES to empty
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        accelerator = Accelerator()
        device = accelerator.device
    else:
        # Handle GPU device specification (e.g., "0,1,2")
        if ',' in args.device:
            # Multiple GPUs specified
            gpu_ids = args.device
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            print(f"Using multiple GPUs: {gpu_ids}")
        else:
            # Single GPU specified
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            print(f"Using GPU: {args.device}")
        
        # Initialize accelerator after setting CUDA_VISIBLE_DEVICES
        accelerator = Accelerator()
        device = accelerator.device
    
    return accelerator, device


def init_log(args):
    args.save_path = os.path.join(
        args.log_path, 
        args.fingerprint_method,
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Log path initialized: {args.save_path}")
    # set log
    log_path = os.path.join(args.save_path, 'log.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
        ]
    )
    logger = logging.getLogger()
    return logger


def save_tensor_dict(tensor_dict, save_path):
    """
    Save a dictionary containing torch.Tensor or list of torch.Tensor to file.
    
    Args:
        tensor_dict (dict): Dictionary with string keys and torch.Tensor or list of torch.Tensor values
        save_path (str): Path to save the dictionary
    """
    if not isinstance(tensor_dict, dict):
        raise TypeError("Input must be a dictionary")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to CPU and save
    save_dict = {}
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Single tensor
            save_dict[key] = value.cpu()
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            # List of tensors
            save_dict[key] = [tensor.cpu() for tensor in value]
        else:
            raise TypeError(f"Value for key '{key}' must be torch.Tensor or list of torch.Tensor")
    
    torch.save(save_dict, save_path)
    print(f"Tensor dictionary saved to: {save_path}")


def load_tensor_dict(load_path, device=None):
    """
    Load a dictionary containing torch.Tensor or list of torch.Tensor from file.
    
    Args:
        load_path (str): Path to load the dictionary from
        device (torch.device, optional): Device to load tensors to. If None, keeps original device.
    
    Returns:
        dict: Dictionary with string keys and torch.Tensor or list of torch.Tensor values
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File {load_path} does not exist")
    
    # Load the dictionary
    loaded_dict = torch.load(load_path, map_location='cpu')
    
    if device is not None:
        # Move tensors to specified device
        result_dict = {}
        for key, value in loaded_dict.items():
            if isinstance(value, torch.Tensor):
                # Single tensor
                result_dict[key] = value.to(device)
            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                # List of tensors
                result_dict[key] = [tensor.to(device) for tensor in value]
            else:
                result_dict[key] = value
        return result_dict
    else:
        return loaded_dict


def save_tensor_dict_with_metadata(tensor_dict, save_path, metadata=None):
    """
    Save a dictionary containing torch.Tensor or list of torch.Tensor with additional metadata.
    
    Args:
        tensor_dict (dict): Dictionary with string keys and torch.Tensor or list of torch.Tensor values
        save_path (str): Path to save the dictionary
        metadata (dict, optional): Additional metadata to save alongside tensors
    """
    if not isinstance(tensor_dict, dict):
        raise TypeError("Input must be a dictionary")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'tensors': {},
        'metadata': metadata or {},
        'save_time': datetime.datetime.now().isoformat()
    }
    
    # Convert tensors to CPU and save
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Single tensor
            save_dict['tensors'][key] = value.cpu()
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            # List of tensors
            save_dict['tensors'][key] = [tensor.cpu() for tensor in value]
        else:
            raise TypeError(f"Value for key '{key}' must be torch.Tensor or list of torch.Tensor")
    
    torch.save(save_dict, save_path)
    print(f"Tensor dictionary with metadata saved to: {save_path}")


def load_tensor_dict_with_metadata(load_path, device=None):
    """
    Load a dictionary containing torch.Tensor or list of torch.Tensor with metadata from file.
    
    Args:
        load_path (str): Path to load the dictionary from
        device (torch.device, optional): Device to load tensors to. If None, keeps original device.
    
    Returns:
        tuple: (tensor_dict, metadata) where tensor_dict contains tensors and metadata contains additional info
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File {load_path} does not exist")
    
    # Load the dictionary
    loaded_dict = torch.load(load_path, map_location='cpu')
    
    # Extract tensors and metadata
    tensor_dict = loaded_dict.get('tensors', {})
    metadata = loaded_dict.get('metadata', {})
    
    if device is not None:
        # Move tensors to specified device
        result_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                # Single tensor
                result_dict[key] = value.to(device)
            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                # List of tensors
                result_dict[key] = [tensor.to(device) for tensor in value]
            else:
                result_dict[key] = value
        return result_dict, metadata
    else:
        return tensor_dict, metadata
    
if __name__ == "__main__":
    path = 'config/benchmark_config.yaml'
    config = load_config(path)
    print(config)