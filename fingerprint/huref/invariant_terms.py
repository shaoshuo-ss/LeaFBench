import re
import numpy as np
import torch
import os
import random
import logging
from PIL import Image
from transformers import AutoModelForCausalLM
from fingerprint.huref.encoder_train import CNNEncoder
import argparse

def make_square_matrix(weight, target_size):
    """
    Make a weight matrix square by repeating it to match the target size.
    For MQA/GQA models where k_proj and v_proj might be smaller than q_proj.
    
    Args:
        weight: Input weight tensor of shape [smaller_dim, hidden_size]
        target_size: Target output dimension to match q_proj
        
    Returns:
        Square weight matrix of shape [target_size, hidden_size]
    """
    if weight.shape[0] == target_size:
        return weight
    
    # Calculate how many times we need to repeat
    repeat_factor = target_size // weight.shape[0]
    remainder = target_size % weight.shape[0]
    
    # Repeat the weight matrix
    repeated_weights = weight.repeat(repeat_factor, 1)
    
    # Handle remainder by taking partial rows
    if remainder > 0:
        partial_weight = weight[:remainder]
        repeated_weights = torch.cat([repeated_weights, partial_weight], dim=0)
    
    return repeated_weights


def detect_model_architecture(state_dict, name):
    """
    Automatically detect model architecture and return model type, layer pattern, and total layers.
    """
    # Check for different model architectures based on parameter names
    sample_keys = list(state_dict.keys())
    
    # Extract layer numbers from all keys
    pattern = r'\.(\d+)\.'
    layer_numbers = set()
    
    for key in sample_keys:
        matches = re.findall(pattern, key)
        for match in matches:
            layer_numbers.add(int(match))
    
    if layer_numbers:
        total_layers = max(layer_numbers) + 1
    else:
        total_layers = 0
    
    # Detect model type based on parameter patterns and model name
    # Check for specific model architectures
    if any('model.layers.' in key for key in sample_keys):
        # Llama-style architectures (Llama, Qwen2.5, Mistral)
        if 'qwen' in name.lower() or 'Qwen' in name:
            return 'qwen2.5', 'model.layers.{}', total_layers
        elif 'llama' in name.lower() or 'Llama' in name:
            return 'llama', 'model.layers.{}', total_layers
        elif 'mistral' in name.lower() or 'Mistral' in name:
            return 'mistral', 'model.layers.{}', total_layers
        else:
            return 'llama', 'model.layers.{}', total_layers  # Default to llama style
    
    elif any('model.decoder.layers.' in key for key in sample_keys):
        # Gemma style
        return 'gemma', 'model.decoder.layers.{}', total_layers
    
    elif any('transformer.h.' in key for key in sample_keys):
        # Phi style
        return 'phi', 'transformer.h.{}', total_layers
    
    elif any('model.layers.' in key for key in sample_keys):
        # Alternative Gemma pattern
        return 'gemma', 'model.layers.{}', total_layers
    
    else:
        # Fallback: analyze model name
        name_lower = name.lower()
        if 'gemma' in name_lower:
            return 'gemma', 'model.layers.{}', total_layers
        elif 'phi' in name_lower:
            return 'phi', 'transformer.h.{}', total_layers
        elif any(model in name_lower for model in ['qwen', 'llama', 'mistral']):
            return 'llama', 'model.layers.{}', total_layers
        else:
            # Ultimate fallback - try to detect from available keys
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not auto-detect architecture for {name}")
            logger.warning(f"Sample keys: {sample_keys[:5]}")
            return 'llama', 'model.layers.{}', total_layers  # Default fallback

def extract_llama_style_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list):
    """
    Extract weights from Llama-style models (Llama, Qwen2.5, Mistral).
    """
    logger = logging.getLogger(__name__)
    
    for layer_idx in target_layers:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Get attention weights
        q_proj = state_dict.get(f"{layer_prefix}.self_attn.q_proj.weight")
        k_proj = state_dict.get(f"{layer_prefix}.self_attn.k_proj.weight") 
        v_proj = state_dict.get(f"{layer_prefix}.self_attn.v_proj.weight")
        o_proj = state_dict.get(f"{layer_prefix}.self_attn.o_proj.weight")
        
        # Get MLP weights
        gate_proj = state_dict.get(f"{layer_prefix}.mlp.gate_proj.weight")
        up_proj = state_dict.get(f"{layer_prefix}.mlp.up_proj.weight")
        down_proj = state_dict.get(f"{layer_prefix}.mlp.down_proj.weight")
        
        if all(w is not None for w in [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]):
            # Ensure all weights are on the same device as x to handle accelerate device distribution
            target_device = x.device
            q_proj_device = q_proj.to(target_device)
            k_proj_device = k_proj.to(target_device)
            v_proj_device = v_proj.to(target_device)
            o_proj_device = o_proj.to(target_device)
            gate_proj_device = gate_proj.to(target_device)
            up_proj_device = up_proj.to(target_device)
            down_proj_device = down_proj.to(target_device)
            
            # Handle MQA/GQA: make k_proj and v_proj square matrices if needed
            target_size = q_proj_device.shape[0]  # Use q_proj size as reference
            if k_proj_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: k_proj shape {k_proj_device.shape} != q_proj shape {q_proj_device.shape}, expanding k_proj")
                k_proj_device = make_square_matrix(k_proj_device, target_size)
            
            if v_proj_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: v_proj shape {v_proj_device.shape} != q_proj shape {q_proj_device.shape}, expanding v_proj")
                v_proj_device = make_square_matrix(v_proj_device, target_size)
            
            # Compute invariant terms
            WqWk = x @ q_proj_device.t() @ k_proj_device @ x.t()
            WqWk_list.append(WqWk)

            WvWo = x @ v_proj_device.t() @ o_proj_device.t() @ x.t()
            WvWo_list.append(WvWo)
            
            WuWd = x @ (gate_proj_device.t() * up_proj_device.t()) @ down_proj_device.t() @ x.t()
            WuWd_list.append(WuWd)
            
            logger.info(f"Extracted weights from layer {layer_idx}")
        else:
            logger.warning(f"Could not find all required weights for layer {layer_idx}")

def extract_gemma_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list):
    """
    Extract weights from Gemma-style models.
    """
    logger = logging.getLogger(__name__)
    
    for layer_idx in target_layers:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Try different Gemma parameter patterns
        # Pattern 1: Standard Gemma
        q_proj = state_dict.get(f"{layer_prefix}.self_attn.q_proj.weight")
        k_proj = state_dict.get(f"{layer_prefix}.self_attn.k_proj.weight")
        v_proj = state_dict.get(f"{layer_prefix}.self_attn.v_proj.weight") 
        o_proj = state_dict.get(f"{layer_prefix}.self_attn.o_proj.weight")
        
        # Pattern 2: Alternative Gemma naming
        if q_proj is None:
            q_proj = state_dict.get(f"{layer_prefix}.attention.q_proj.weight")
            k_proj = state_dict.get(f"{layer_prefix}.attention.k_proj.weight")
            v_proj = state_dict.get(f"{layer_prefix}.attention.v_proj.weight")
            o_proj = state_dict.get(f"{layer_prefix}.attention.o_proj.weight")
        
        # Get MLP weights - try different patterns
        gate_proj = state_dict.get(f"{layer_prefix}.mlp.gate_proj.weight")
        up_proj = state_dict.get(f"{layer_prefix}.mlp.up_proj.weight")
        down_proj = state_dict.get(f"{layer_prefix}.mlp.down_proj.weight")
        
        # Alternative MLP naming
        if gate_proj is None:
            gate_proj = state_dict.get(f"{layer_prefix}.feed_forward.gate_proj.weight")
            up_proj = state_dict.get(f"{layer_prefix}.feed_forward.up_proj.weight")
            down_proj = state_dict.get(f"{layer_prefix}.feed_forward.down_proj.weight")
        
        if all(w is not None for w in [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]):
            # Ensure all weights are on the same device as x to handle accelerate device distribution
            target_device = x.device
            q_proj_device = q_proj.to(target_device)
            k_proj_device = k_proj.to(target_device)
            v_proj_device = v_proj.to(target_device)
            o_proj_device = o_proj.to(target_device)
            gate_proj_device = gate_proj.to(target_device)
            up_proj_device = up_proj.to(target_device)
            down_proj_device = down_proj.to(target_device)
            
            # Handle MQA/GQA: make k_proj and v_proj square matrices if needed
            target_size = q_proj_device.shape[0]  # Use q_proj size as reference
            if k_proj_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: k_proj shape {k_proj_device.shape} != q_proj shape {q_proj_device.shape}, expanding k_proj")
                k_proj_device = make_square_matrix(k_proj_device, target_size)
            
            if v_proj_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: v_proj shape {v_proj_device.shape} != q_proj shape {q_proj_device.shape}, expanding v_proj")
                v_proj_device = make_square_matrix(v_proj_device, target_size)
            
            # Compute invariant terms (same as Llama style)
            WqWk = x @ q_proj_device.t() @ k_proj_device @ x.t()
            WqWk_list.append(WqWk)
            
            WvWo = x @ v_proj_device.t() @ o_proj_device.t() @ x.t()
            WvWo_list.append(WvWo)
            
            WuWd = x @ (gate_proj_device.t() * up_proj_device.t()) @ down_proj_device.t() @ x.t()
            WuWd_list.append(WuWd)
            
            logger.info(f"Extracted weights from layer {layer_idx}")
        else:
            logger.warning(f"Could not find all required weights for layer {layer_idx}")
            missing = []
            if q_proj is None: missing.append("q_proj")
            if k_proj is None: missing.append("k_proj")
            if v_proj is None: missing.append("v_proj")
            if o_proj is None: missing.append("o_proj")
            if gate_proj is None: missing.append("gate_proj")
            if up_proj is None: missing.append("up_proj")
            if down_proj is None: missing.append("down_proj")
            logger.warning(f"Missing weights: {missing}")

def extract_phi_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list):
    """
    Extract weights from Phi-style models.
    """
    logger = logging.getLogger(__name__)
    
    for layer_idx in target_layers:
        layer_prefix = layer_pattern.format(layer_idx)
        
        # Phi models have different parameter naming patterns
        # Try different possible patterns
        q_weight = k_weight = v_weight = o_proj = fc1 = fc2 = None
        
        # Pattern 1: Packed QKV weights
        qkv_weight = state_dict.get(f"{layer_prefix}.mixer.Wqkv.weight")
        if qkv_weight is not None:
            # For MQA/GQA, the packed weight might not be evenly divisible by 3
            # Try to detect the actual split sizes
            total_size = qkv_weight.shape[0]
            
            # Check if it's a standard split (equal sizes)
            if total_size % 3 == 0:
                hidden_size = total_size // 3
                q_weight = qkv_weight[:hidden_size]
                k_weight = qkv_weight[hidden_size:2*hidden_size]
                v_weight = qkv_weight[2*hidden_size:]
            else:
                # For MQA/GQA, we need to detect the actual sizes
                # This is a heuristic - typically q_size is larger than k_size and v_size
                # We'll try to find the pattern by checking parameter names or making educated guesses
                
                # Try to find separate weights to determine sizes
                separate_q = state_dict.get(f"{layer_prefix}.mixer.q_proj.weight")
                separate_k = state_dict.get(f"{layer_prefix}.mixer.k_proj.weight")
                separate_v = state_dict.get(f"{layer_prefix}.mixer.v_proj.weight")
                
                if separate_q is not None and separate_k is not None and separate_v is not None:
                    # Use separate weights if available
                    q_weight = separate_q
                    k_weight = separate_k
                    v_weight = separate_v
                else:
                    # Fallback: assume equal split and let make_square_matrix handle it
                    hidden_size = total_size // 3
                    q_weight = qkv_weight[:hidden_size]
                    k_weight = qkv_weight[hidden_size:2*hidden_size]
                    v_weight = qkv_weight[2*hidden_size:]
        else:
            # Pattern 2: Separate Q, K, V weights
            q_weight = state_dict.get(f"{layer_prefix}.mixer.q_proj.weight")
            k_weight = state_dict.get(f"{layer_prefix}.mixer.k_proj.weight")
            v_weight = state_dict.get(f"{layer_prefix}.mixer.v_proj.weight")
            
            # Pattern 3: Alternative naming
            if q_weight is None:
                q_weight = state_dict.get(f"{layer_prefix}.attn.q_proj.weight")
                k_weight = state_dict.get(f"{layer_prefix}.attn.k_proj.weight")
                v_weight = state_dict.get(f"{layer_prefix}.attn.v_proj.weight")
        
        # Output projection
        o_proj = (state_dict.get(f"{layer_prefix}.mixer.out_proj.weight") or 
                 state_dict.get(f"{layer_prefix}.attn.o_proj.weight"))
        
        # MLP weights - try different patterns
        fc1 = (state_dict.get(f"{layer_prefix}.mlp.fc1.weight") or
               state_dict.get(f"{layer_prefix}.mlp.gate_proj.weight") or
               state_dict.get(f"{layer_prefix}.mlp.up_proj.weight"))
        
        fc2 = (state_dict.get(f"{layer_prefix}.mlp.fc2.weight") or
               state_dict.get(f"{layer_prefix}.mlp.down_proj.weight"))
        
        if all(w is not None for w in [q_weight, k_weight, v_weight, o_proj, fc1, fc2]):
            # Ensure all weights are on the same device as x to handle accelerate device distribution
            target_device = x.device
            q_weight_device = q_weight.to(target_device)
            k_weight_device = k_weight.to(target_device)
            v_weight_device = v_weight.to(target_device)
            o_proj_device = o_proj.to(target_device)
            fc1_device = fc1.to(target_device)
            fc2_device = fc2.to(target_device)
            
            # Handle MQA/GQA: make k_weight and v_weight square matrices if needed
            target_size = q_weight_device.shape[0]  # Use q_weight size as reference
            if k_weight_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: k_weight shape {k_weight_device.shape} != q_weight shape {q_weight_device.shape}, expanding k_weight")
                k_weight_device = make_square_matrix(k_weight_device, target_size)
            
            if v_weight_device.shape[0] != target_size:
                logger.info(f"Layer {layer_idx}: v_weight shape {v_weight_device.shape} != q_weight shape {q_weight_device.shape}, expanding v_weight")
                v_weight_device = make_square_matrix(v_weight_device, target_size)
            
            # Compute invariant terms
            WqWk = x @ q_weight_device.t() @ k_weight_device @ x.t()
            WqWk_list.append(WqWk)
            
            WvWo = x @ v_weight_device.t() @ o_proj_device.t() @ x.t()
            WvWo_list.append(WvWo)
            
            WuWd = x @ fc1_device.t() @ fc2_device.t() @ x.t()
            WuWd_list.append(WuWd)
            
            logger.info(f"Extracted weights from layer {layer_idx}")
        else:
            logger.warning(f"Could not find all required weights for layer {layer_idx}")
            missing = []
            if q_weight is None: missing.append("q_proj")
            if k_weight is None: missing.append("k_proj") 
            if v_weight is None: missing.append("v_proj")
            if o_proj is None: missing.append("o_proj")
            if fc1 is None: missing.append("fc1/gate_proj")
            if fc2 is None: missing.append("fc2/down_proj")
            logger.warning(f"Missing weights: {missing}")

def get_invariant_terms(state_dict, name, selected_tokens):
    """
    Extract invariant terms from model state dict for fingerprinting.
    Automatically detects model architecture and extracts weights from last 2 layers.
    """
    WqWk_list = []
    WvWo_list = []
    WuWd_list = []

    logger = logging.getLogger(__name__)

    logger.info(f"Processing model: {name}")
    logger.info(f"State dict has {len(state_dict)} parameters")
    
    # Check if this is a quantized model
    has_quantized_weights = any('qweight' in key for key in state_dict.keys())
    if has_quantized_weights:
        logger.info(f"Detected quantized model: {name}")

    # Find embedding weights automatically
    x = None
    embedding_keys = [
        'model.embed_tokens.weight',  # Llama, Qwen2.5, Mistral
        'embed_tokens.weight',        # Some variants
        'embeddings.word_embeddings.weight',  # Gemma
        'model.decoder.embed_tokens.weight',  # Some decoder models
        'transformer.wte.weight',     # GPT-style
        'wte.weight'                  # GPT variants
    ]
    
    for key in embedding_keys:
        if key in state_dict:
            x = state_dict[key]
            logger.info(f"Found embedding weights: {key}")
            break
    
    if x is None:
        raise ValueError(f"Could not find embedding weights for model {name}")
    
    x = x[selected_tokens]
    logger.info(f"Selected {len(selected_tokens)} tokens from embedding matrix")
    
    # Auto-detect model architecture and extract last 2 layers
    model_type, layer_pattern, total_layers = detect_model_architecture(state_dict, name)
    logger.info(f"Detected model type: {model_type}, Total layers: {total_layers}")
    
    # Get last 2 layers
    target_layers = [total_layers - 2, total_layers - 1]
    logger.info(f"Extracting weights from layers: {target_layers}")
    
    # Extract weights based on detected architecture
    if model_type in ['llama', 'qwen2.5', 'mistral']:
        extract_llama_style_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list)
    elif model_type == 'gemma':
        extract_gemma_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list)
    elif model_type == 'phi':
        extract_phi_weights(state_dict, layer_pattern, target_layers, x, WqWk_list, WvWo_list, WuWd_list)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Combine all invariant terms
    if len(WqWk_list) == 0:
        raise ValueError(f"Could not extract any weights from model {name}")
    
    logger.info(f"Successfully extracted {len(WqWk_list)} sets of invariant terms")
    
    # Stack and concatenate all invariant terms
    parameters = [torch.stack((t1, t2, t3)) for t1, t2, t3 in zip(WqWk_list, WvWo_list, WuWd_list)]
    parameters = torch.cat(parameters, dim=0)
    
    # Save invariant terms
    # invariant_terms_saved_path = globals().get('invariant_terms_saved_path', './')
    # np.save(os.path.join(invariant_terms_saved_path, f'{str(name)}.npy'), parameters.detach().cpu().numpy())
    
    return parameters

class MeanPooling(torch.nn.Module):
    """
    Mean pooling module that processes invariant terms tensor.
    
    Our target is to perform mean pooling on the input tensor of shape [6, 4096, 4096]
    such that each [6, 4096, 8] block is reduced to a mean value.
    We should end up with a tensor of shape [6, 4096, 512] that is
    then reshaped/flattened to [6*4096*512/4096] = [512].
    """
    
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, input_tensor):
        # Check if the input tensor has the correct shape
        if input_tensor.shape != (6, 4096, 4096):
            raise ValueError('Input tensor must be of shape [6, 4096, 4096]')
        
        reshaped_tensor = input_tensor.view(512, -1)

        # Perform mean pooling over the last dimension
        pooled_tensor = reshaped_tensor.mean(-1)

        # Flatten the tensor to get a vector of shape [512]
        output_vector = pooled_tensor.view(-1)

        # Normalize the output vector
        mean = torch.mean(output_vector)
        std = torch.std(output_vector)
        output_vector = (output_vector - mean) / std
        return output_vector

class CNNEncode(torch.nn.Module):
    """
    CNN encoding module that processes invariant terms using a pre-trained CNN encoder.
    """
    
    def __init__(self, encoder_path):
        super(CNNEncode, self).__init__()
        self.encoder_path = encoder_path
        self.cnn_encoder = None
        self._load_encoder()
    
    def _load_encoder(self):
        """Load the pre-trained CNN encoder"""
        # self.cnn_encoder = CNNEncoder().cuda()
        self.cnn_encoder = torch.load(self.encoder_path)
        self.cnn_encoder.eval()
    
    def forward(self, invariant_terms):
        """
        Forward pass through the CNN encoder.
        
        Args:
            invariant_terms: Input tensor of invariant terms
            
        Returns:
            output_vector: Encoded feature vector
        """
        # Input normalization
        normalized_terms = invariant_terms.clone()
        for i in range(normalized_terms.shape[0]):
            mean = torch.mean(normalized_terms[i])
            std = torch.std(normalized_terms[i])
            normalized_terms[i] = (normalized_terms[i] - mean) / std
        
        # Pass through CNN encoder
        output_vector = self.cnn_encoder(normalized_terms.unsqueeze(0).unsqueeze(0).cuda())
        
        # Output normalization
        mean = torch.mean(output_vector)
        std = torch.std(output_vector)
        output_vector = (output_vector - mean) / std
        
        return output_vector.squeeze(0)
