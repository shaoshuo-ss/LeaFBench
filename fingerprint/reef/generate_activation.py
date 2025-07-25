import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import pandas as pd
from tqdm import tqdm

ACTS_BATCH_SIZE = 400


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out = module_outputs 

def load_statements(dataset_path):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(dataset_path)
    statements = dataset['statement'].tolist()
    return statements

def get_model_layer_count(model, model_name):
    """
    Get the total number of layers for the given model.
    """
    if 'mpt-7b' in model_name.lower():
        return len(model.transformer.blocks)
    elif 'falcon-7b' in model_name.lower():
        return len(model.transformer.h)
    elif any(name in model_name.lower() for name in ['qwen2.5', 'qwen-2.5', 'llama-3.1', 'llama3.1', 'mistral-7b', 'gemma-2', 'phi-4']):
        return len(model.model.layers)
    else:
        # Default case: try to get layer count from model.model.layers
        try:
            return len(model.model.layers)
        except AttributeError:
            raise ValueError(f"Unsupported model architecture for {model_name}")

def parse_layer_spec(layer_spec, total_layers):
    """
    Parse layer specification and return list of layer indices.
    
    Args:
        layer_spec: Can be:
            - int: single layer index
            - list of ints: multiple layer indices
            - str: layer range like "0-5" or "last-3" 
            - "all": all layers
        total_layers: total number of layers in the model
    
    Returns:
        List of layer indices
    """
    if isinstance(layer_spec, int):
        return [layer_spec]
    elif isinstance(layer_spec, list):
        return layer_spec
    elif isinstance(layer_spec, str):
        if layer_spec == "all":
            return list(range(total_layers))
        elif layer_spec.startswith("last-"):
            num_layers = int(layer_spec.split("-")[1])
            return list(range(max(0, total_layers - num_layers), total_layers))
        elif "-" in layer_spec:
            start, end = map(int, layer_spec.split("-"))
            return list(range(start, min(end + 1, total_layers)))
        else:
            return [int(layer_spec)]
    else:
        raise ValueError(f"Invalid layer specification: {layer_spec}")

def get_layer_module(model, model_name, layer_idx):
    """
    Get the specific layer module for registering hooks based on model architecture.
    """
    model_name_lower = model_name.lower()
    
    if 'mpt-7b' in model_name_lower:
        return model.transformer.blocks[layer_idx]
    elif 'falcon-7b' in model_name_lower:
        return model.transformer.h[layer_idx]
    elif any(name in model_name_lower for name in ['qwen2.5', 'qwen-2.5']):
        return model.model.layers[layer_idx]
    elif any(name in model_name_lower for name in ['llama-3.1', 'llama3.1']):
        return model.model.layers[layer_idx]
    elif 'mistral-7b' in model_name_lower:
        return model.model.layers[layer_idx]
    elif 'gemma-2' in model_name_lower:
        return model.model.layers[layer_idx]
    elif 'phi-4' in model_name_lower:
        return model.model.layers[layer_idx]
    else:
        # Default case: try model.model.layers
        try:
            return model.model.layers[layer_idx]
        except AttributeError:
            raise ValueError(f"Unsupported model architecture for {model_name}")

def get_acts(statements, tokenizer, model, model_name, layers, device, token_pos=-1, batch_size=1):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.

    Args:
        statements: List of input statements
        tokenizer: Model tokenizer
        model: The model to extract activations from
        model_name: Name of the model (used for architecture detection)
        layers: Layer specification (int, list, str like "0-5", "last-3", "all")
        device: Device to run on
        token_pos: Position of token to extract activations from (default: -1 for last token)
        batch_size: Number of statements to process in each batch (default: 1)
    
    Returns:
        Dictionary mapping layer indices to stacked activation tensors
    """
    # Get total number of layers and parse layer specification
    total_layers = get_model_layer_count(model, model_name)
    layer_indices = parse_layer_spec(layers, total_layers)
    
    print(f"Model {model_name} has {total_layers} layers")
    print(f"Extracting activations from layers: {layer_indices}")
    
    # attach hooks
    hooks, handles = [], []
    for layer_idx in layer_indices:
        hook = Hook()
        layer_module = get_layer_module(model, model_name, layer_idx)
        handle = layer_module.register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)
    
    # get activations
    acts = {layer_idx: [] for layer_idx in layer_indices}
    
    # Process statements in batches
    for i in tqdm(range(0, len(statements), batch_size), desc="Extracting activations"):
        batch_statements = statements[i:i + batch_size]
        
        # Tokenize the entire batch
        batch_inputs = tokenizer(batch_statements, return_tensors="pt", padding=True, truncation=True)
        batch_inputs = {k: v.to(device=model.device) for k, v in batch_inputs.items()}
        
        # Forward pass for the entire batch
        with torch.no_grad():
            model(**batch_inputs)
        
        # Extract activations from each hook for the batch
        for layer_idx, hook in zip(layer_indices, hooks):
            # hook.out shape: [batch_size, seq_len, hidden_size]
            batch_acts = hook.out[0][:, token_pos]  # [batch_size, hidden_size]
            acts[layer_idx].extend(batch_acts)
    
    # stack len(statements)'s activations
    for layer_idx, act in acts.items():
        acts[layer_idx] = torch.stack(act).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts
