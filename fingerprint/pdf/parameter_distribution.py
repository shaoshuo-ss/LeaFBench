import torch
import torch.nn as nn
import logging
import numpy as np
from scipy.stats import pearsonr


def _detect_model_architecture(state_dict, model_name=None):
    """
    Detect model architecture based on state dict keys.
    
    Args:
        state_dict: Model state dictionary
        model_name: Optional model name for additional hints
        
    Returns:
        tuple: (architecture_type, layer_pattern, total_layers)
    """
    sample_keys = list(state_dict.keys())
    
    # Extract layer numbers
    import re
    pattern = r'\.(\d+)\.'
    layer_numbers = set()
    
    for key in sample_keys:
        matches = re.findall(pattern, key)
        for match in matches:
            layer_numbers.add(int(match))
    
    total_layers = max(layer_numbers) + 1 if layer_numbers else 0
    
    # Detect architecture
    if any('model.layers.' in key for key in sample_keys):
        # Llama-style (Qwen2.5, Llama, Mistral)
        return 'llama_style', 'model.layers.{}', total_layers
    elif any('model.decoder.layers.' in key for key in sample_keys):
        # Gemma style
        return 'gemma', 'model.decoder.layers.{}', total_layers
    elif any('transformer.h.' in key for key in sample_keys):
        # Phi style
        return 'phi', 'transformer.h.{}', total_layers
    else:
        # Default fallback
        return 'llama_style', 'model.layers.{}', total_layers


def _extract_llama_style_weights(state_dict, layer_pattern, layer_indices):
    """
    Extract weights from Llama-style models (Qwen2.5, Llama, Mistral).
    
    Args:
        state_dict: Model state dictionary
        layer_pattern: Layer naming pattern (e.g., 'model.layers.{}')
        layer_indices: List of layer indices to extract
        
    Returns:
        tuple: (W_q_stds, W_k_stds, W_v_stds, W_o_stds) - std for each layer
    """
    W_q_stds = []
    W_k_stds = []
    W_v_stds = []
    W_o_stds = []
    
    for layer_idx in layer_indices:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Get attention weights
        q_proj = state_dict.get(f"{layer_prefix}.self_attn.q_proj.weight")
        k_proj = state_dict.get(f"{layer_prefix}.self_attn.k_proj.weight")
        v_proj = state_dict.get(f"{layer_prefix}.self_attn.v_proj.weight")
        o_proj = state_dict.get(f"{layer_prefix}.self_attn.o_proj.weight")
        
        if all(w is not None for w in [q_proj, k_proj, v_proj, o_proj]):
            W_q_stds.append(torch.std(q_proj).item())
            W_k_stds.append(torch.std(k_proj).item())
            W_v_stds.append(torch.std(v_proj).item())
            W_o_stds.append(torch.std(o_proj).item())
        else:
            # Handle missing weights
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing attention weights for layer {layer_idx}")
            W_q_stds.append(0.0)
            W_k_stds.append(0.0)
            W_v_stds.append(0.0)
            W_o_stds.append(0.0)
    
    return W_q_stds, W_k_stds, W_v_stds, W_o_stds


def _extract_gemma_weights(state_dict, layer_pattern, layer_indices):
    """
    Extract weights from Gemma-style models.
    """
    W_q_stds = []
    W_k_stds = []
    W_v_stds = []
    W_o_stds = []
    
    for layer_idx in layer_indices:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Try different naming patterns for Gemma
        q_proj = (state_dict.get(f"{layer_prefix}.self_attn.q_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attention.q_proj.weight"))
        k_proj = (state_dict.get(f"{layer_prefix}.self_attn.k_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attention.k_proj.weight"))
        v_proj = (state_dict.get(f"{layer_prefix}.self_attn.v_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attention.v_proj.weight"))
        o_proj = (state_dict.get(f"{layer_prefix}.self_attn.o_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attention.o_proj.weight"))
        
        if all(w is not None for w in [q_proj, k_proj, v_proj, o_proj]):
            W_q_stds.append(torch.std(q_proj).item())
            W_k_stds.append(torch.std(k_proj).item())
            W_v_stds.append(torch.std(v_proj).item())
            W_o_stds.append(torch.std(o_proj).item())
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing attention weights for layer {layer_idx}")
            W_q_stds.append(0.0)
            W_k_stds.append(0.0)
            W_v_stds.append(0.0)
            W_o_stds.append(0.0)
    
    return W_q_stds, W_k_stds, W_v_stds, W_o_stds


def _extract_phi_weights(state_dict, layer_pattern, layer_indices):
    """
    Extract weights from Phi-style models.
    """
    W_q_stds = []
    W_k_stds = []
    W_v_stds = []
    W_o_stds = []
    
    for layer_idx in layer_indices:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Try different patterns for Phi models
        q_weight = k_weight = v_weight = o_proj = None
        
        # Pattern 1: Packed QKV weights
        qkv_weight = state_dict.get(f"{layer_prefix}.mixer.Wqkv.weight")
        if qkv_weight is not None:
            # Split QKV weights - assume equal split
            total_size = qkv_weight.shape[0]
            if total_size % 3 == 0:
                hidden_size = total_size // 3
                q_weight = qkv_weight[:hidden_size]
                k_weight = qkv_weight[hidden_size:2*hidden_size]
                v_weight = qkv_weight[2*hidden_size:]
        
        # Pattern 2: Separate Q, K, V weights
        if q_weight is None:
            q_weight = (state_dict.get(f"{layer_prefix}.mixer.q_proj.weight") or
                       state_dict.get(f"{layer_prefix}.attn.q_proj.weight"))
            k_weight = (state_dict.get(f"{layer_prefix}.mixer.k_proj.weight") or
                       state_dict.get(f"{layer_prefix}.attn.k_proj.weight"))
            v_weight = (state_dict.get(f"{layer_prefix}.mixer.v_proj.weight") or
                       state_dict.get(f"{layer_prefix}.attn.v_proj.weight"))
        
        # Output projection
        o_proj = (state_dict.get(f"{layer_prefix}.mixer.out_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attn.o_proj.weight"))
        
        if all(w is not None for w in [q_weight, k_weight, v_weight, o_proj]):
            W_q_stds.append(torch.std(q_weight).item())
            W_k_stds.append(torch.std(k_weight).item())
            W_v_stds.append(torch.std(v_weight).item())
            W_o_stds.append(torch.std(o_proj).item())
        else:
            logger = logging.getLogger(__name__)
            logger.warning(f"Missing attention weights for layer {layer_idx}")
            W_q_stds.append(0.0)
            W_k_stds.append(0.0)
            W_v_stds.append(0.0)
            W_o_stds.append(0.0)
    
    return W_q_stds, W_k_stds, W_v_stds, W_o_stds


def get_transformer_parameters(model):
    """
    Extract transformer parameters (W_q, W_k, W_v, W_o) standard deviations from each layer
    and normalize them to have zero mean and unit variance.
    
    Args:
        model: A transformers AutoModelForCausalLM model
        
    Returns:
        tuple: (Sq, Sk, Sv, So) - normalized tensors of standard deviations for each weight type
    """
    logger = logging.getLogger(__name__)
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Detect model architecture
    model_name = getattr(model, 'name_or_path', '') or str(type(model))
    arch_type, layer_pattern, total_layers = _detect_model_architecture(state_dict, model_name)
    
    logger.info(f"Detected architecture: {arch_type}, total layers: {total_layers}")
    
    # Extract weights based on architecture
    layer_indices = list(range(total_layers))
    
    if arch_type == 'llama_style':
        W_q_stds, W_k_stds, W_v_stds, W_o_stds = _extract_llama_style_weights(
            state_dict, layer_pattern, layer_indices)
    elif arch_type == 'gemma':
        W_q_stds, W_k_stds, W_v_stds, W_o_stds = _extract_gemma_weights(
            state_dict, layer_pattern, layer_indices)
    elif arch_type == 'phi':
        W_q_stds, W_k_stds, W_v_stds, W_o_stds = _extract_phi_weights(
            state_dict, layer_pattern, layer_indices)
    else:
        raise ValueError(f"Unsupported architecture: {arch_type}")
    
    # Convert to tensors
    Sq = torch.tensor(W_q_stds, dtype=torch.float32)
    Sk = torch.tensor(W_k_stds, dtype=torch.float32)
    Sv = torch.tensor(W_v_stds, dtype=torch.float32)
    So = torch.tensor(W_o_stds, dtype=torch.float32)
    
    # Normalize to zero mean and unit variance
    def normalize_tensor(tensor):
        if len(tensor) <= 1:
            return tensor
        mean = torch.mean(tensor)
        std = torch.std(tensor, unbiased=False)
        if std == 0:
            return tensor - mean  # Just center if no variance
        return (tensor - mean) / std
    
    Sq = normalize_tensor(Sq)
    Sk = normalize_tensor(Sk)
    Sv = normalize_tensor(Sv)
    So = normalize_tensor(So)
    
    logger.info(f"Extracted parameter distributions: Sq shape {Sq.shape}, Sk shape {Sk.shape}, "
               f"Sv shape {Sv.shape}, So shape {So.shape}")
    
    return Sq, Sk, Sv, So


def get_correlation_coefficient(param1, param2):
    """
    Calculate the Pearson correlation coefficient between two parameter tensors.
    If the tensors have different lengths, the shorter one will be linearly interpolated
    to match the length of the longer one.
    
    Args:
        param1: First parameter tensor
        param2: Second parameter tensor
        
    Returns:
        float: Pearson correlation coefficient between the two tensors
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(param1):
        param1 = param1.detach().cpu().numpy()
    if torch.is_tensor(param2):
        param2 = param2.detach().cpu().numpy()
    
    # Flatten tensors
    param1_flat = param1.flatten()
    param2_flat = param2.flatten()
    
    # Check if we have enough data points
    if len(param1_flat) < 2 and len(param2_flat) < 2:
        return 0.0  # Cannot calculate correlation with less than 2 points
    
    # Handle case where one tensor is much smaller
    if len(param1_flat) < 2:
        # If param1 has only one element, replicate it to match param2 length
        param1_flat = np.full(len(param2_flat), param1_flat[0])
    elif len(param2_flat) < 2:
        # If param2 has only one element, replicate it to match param1 length
        param2_flat = np.full(len(param1_flat), param2_flat[0])
    
    # If tensors have different lengths, interpolate the shorter one
    if len(param1_flat) != len(param2_flat):
        if len(param1_flat) < len(param2_flat):
            # Interpolate param1 to match param2 length
            target_length = len(param2_flat)
            shorter_seq = param1_flat
            longer_seq = param2_flat
            
            # Create interpolation indices
            old_indices = np.linspace(0, len(shorter_seq) - 1, len(shorter_seq))
            new_indices = np.linspace(0, len(shorter_seq) - 1, target_length)
            
            # Perform linear interpolation
            param1_flat = np.interp(new_indices, old_indices, shorter_seq)
            param2_flat = longer_seq
            
        else:
            # Interpolate param2 to match param1 length
            target_length = len(param1_flat)
            shorter_seq = param2_flat
            longer_seq = param1_flat
            
            # Create interpolation indices
            old_indices = np.linspace(0, len(shorter_seq) - 1, len(shorter_seq))
            new_indices = np.linspace(0, len(shorter_seq) - 1, target_length)
            
            # Perform linear interpolation
            param2_flat = np.interp(new_indices, old_indices, shorter_seq)
            param1_flat = longer_seq
    
    # Calculate correlation coefficient
    if len(param1_flat) < 2:
        return 0.0  # Still cannot calculate correlation
    
    correlation, _ = pearsonr(param1_flat, param2_flat)
    
    # Handle NaN case
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)
