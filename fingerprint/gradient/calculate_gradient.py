import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import skew, kurtosis
from collections import defaultdict


def load_statements(dataset_path):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(dataset_path)
    statements = dataset['statement'].tolist()
    return statements


def _get_layer_name_mapping(model):
    """
    Get layer name mapping for different model architectures.
    
    Returns:
        dict: Mapping of layer types to parameter name patterns
    """
    model_name = model.__class__.__name__.lower()
    
    # Common patterns for different model architectures
    layer_patterns = {
        'attention': [],
        'ffn': [],
        'embedding': []
    }
    
    # Qwen2.5 models
    if 'qwen' in model_name:
        layer_patterns['attention'] = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        layer_patterns['ffn'] = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        layer_patterns['embedding'] = ['embed_tokens', 'lm_head']
    
    # Llama models
    elif 'llama' in model_name:
        layer_patterns['attention'] = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        layer_patterns['ffn'] = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        layer_patterns['embedding'] = ['embed_tokens', 'lm_head']
    
    # Mistral models
    elif 'mistral' in model_name:
        layer_patterns['attention'] = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        layer_patterns['ffn'] = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        layer_patterns['embedding'] = ['embed_tokens', 'lm_head']
    
    # Gemma models
    elif 'gemma' in model_name:
        layer_patterns['attention'] = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        layer_patterns['ffn'] = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        layer_patterns['embedding'] = ['embed_tokens', 'lm_head']
    
    # Phi models
    elif 'phi' in model_name:
        layer_patterns['attention'] = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.dense']
        layer_patterns['ffn'] = ['mlp.fc1', 'mlp.fc2']
        layer_patterns['embedding'] = ['embed_tokens', 'lm_head']
    
    # Default fallback for transformers
    else:
        # Try common transformer patterns
        layer_patterns['attention'] = ['attn', 'attention', 'self_attn']
        layer_patterns['ffn'] = ['mlp', 'ffn', 'feed_forward']
        layer_patterns['embedding'] = ['embed', 'embedding', 'lm_head', 'head']
    
    return layer_patterns


def _categorize_gradients(model, gradients):
    """
    Categorize gradients by layer type (attention, FFN, embedding).
    
    Args:
        model: The model
        gradients: Dict of parameter name -> gradient tensor
        
    Returns:
        dict: Categorized gradients
    """
    layer_patterns = _get_layer_name_mapping(model)
    
    categorized = {
        'attention': [],
        'ffn': [],
        'embedding': [],
        'all': []
    }
    
    for param_name, grad in gradients.items():
        if grad is not None:
            categorized['all'].append(grad.flatten())
            
            # Check which category this parameter belongs to
            param_categorized = False
            
            for category, patterns in layer_patterns.items():
                for pattern in patterns:
                    if pattern in param_name:
                        categorized[category].append(grad.flatten())
                        param_categorized = True
                        break
                if param_categorized:
                    break
    
    # Convert lists to tensors
    for category in categorized:
        if categorized[category]:
            categorized[category] = torch.cat(categorized[category])
        else:
            # If no gradients found for this category, create empty tensor
            categorized[category] = torch.tensor([])
    
    return categorized


def _compute_gradient_stats(grad_tensor):
    """
    Compute statistical properties of gradients.
    
    Args:
        grad_tensor: Flattened gradient tensor
        
    Returns:
        tuple: (mean, variance, l2_norm, skewness, kurtosis)
    """
    if len(grad_tensor) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Convert to numpy for statistical calculations
    grad_np = grad_tensor.detach().cpu().numpy()
    
    mean = float(np.mean(grad_np))
    variance = float(np.var(grad_np))
    l2_norm = float(torch.norm(grad_tensor, p=2).item())
    
    # Avoid division by zero for skewness and kurtosis
    if variance > 1e-10:
        skewness = float(skew(grad_np))
        kurt = float(kurtosis(grad_np))
    else:
        skewness = 0.0
        kurt = 0.0
    
    return mean, variance, l2_norm, skewness, kurt


def _count_model_info(model):
    """
    Count total parameters and layers in the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        tuple: (total_params, total_layers)
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count layers - look for common layer patterns
    total_layers = 0
    for name, module in model.named_modules():
        # Count transformer layers/blocks
        if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'decoder']):
            if name.count('.') == 1:  # Only count top-level layers
                total_layers += 1
    
    # If no layers found with the above method, try alternative approach
    if total_layers == 0:
        for name, module in model.named_modules():
            if 'layers' in name and name.endswith('.layers'):
                # Count the number of layers in the layers module
                layers_module = module
                total_layers = len(layers_module) if hasattr(layers_module, '__len__') else 0
                break
    
    return total_params, total_layers


def get_gradients_stats(model, tokenizer, statements, batch_size=1):
    """
    Get gradients statistics for the model on the given statements.
    
    Args:
        model: The model to analyze (AutoModelForCausalLM).
        tokenizer: The tokenizer for the model.
        statements (list): List of input statements.
        batch_size (int): Batch size for processing.

    Returns:
        torch.Tensor: A tensor containing gradient statistics:
            [0-4]: All gradients (mean, var, L2, skew, kurt)
            [5-7]: Attention gradients (mean, var, L2)
            [8-10]: FFN gradients (mean, var, L2) 
            [11-13]: Embedding gradients (mean, var, L2)
            [14-18]: Relative statistics (attn_ratio, ffn_ratio, emb_ratio, attn_l2_ratio, ffn_l2_ratio)
    """
    # Ensure model is in the right state
    model.train()  # Set model to training mode for gradient computation
    device = next(model.parameters()).device
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize accumulators for gradient statistics (not gradients themselves)
    gradient_stats_accumulator = defaultdict(lambda: {
        'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0
    })
    
    num_batches = len(range(0, len(statements), batch_size))
    
    # Process statements in batches
    for i in range(0, len(statements), batch_size):
        batch_statements = statements[i:i + batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch_statements,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradient statistics directly (without storing gradients)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                
                # Use double precision for better numerical stability
                grad_flat_double = grad_flat.double()
                
                # Compute statistics with overflow protection but maintain precision
                grad_sum = grad_flat_double.sum().item()
                grad_abs_sum = grad_flat_double.abs().sum().item()
                
                # Compute sum of squares with better precision
                grad_sq = grad_flat_double ** 2
                grad_sum_sq = grad_sq.sum().item()
                
                # Check for overflow/underflow with more conservative bounds
                if abs(grad_sum) > 1e20:
                    grad_sum = np.sign(grad_sum) * 1e20
                if grad_sum_sq > 1e20:
                    grad_sum_sq = 1e20
                if grad_abs_sum > 1e20:
                    grad_abs_sum = 1e20
                
                # Accumulate statistics
                gradient_stats_accumulator[name]['sum'] += grad_sum
                gradient_stats_accumulator[name]['sum_sq'] += grad_sum_sq
                gradient_stats_accumulator[name]['sum_abs'] += grad_abs_sum
                gradient_stats_accumulator[name]['count'] += grad_flat.numel()
        
        # Free memory immediately after each batch
        del inputs, outputs, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute final averaged gradients from accumulated statistics
    final_gradients = {}
    for name, stats in gradient_stats_accumulator.items():
        if stats['count'] > 0:
            # Reconstruct mean gradient (approximation for categorization)
            mean_grad = stats['sum'] / stats['count']
            # Create a tensor with the mean value for categorization purposes
            # This is much smaller than storing full gradients
            final_gradients[name] = torch.tensor([mean_grad], device='cpu')
    
    # Categorize parameters by layer type for later statistics computation
    layer_patterns = _get_layer_name_mapping(model)
    categorized_stats = {
        'attention': {'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0},
        'ffn': {'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0},
        'embedding': {'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0},
        'all': {'sum': 0.0, 'sum_sq': 0.0, 'sum_abs': 0.0, 'count': 0}
    }
    
    # Aggregate statistics by category
    for name, stats in gradient_stats_accumulator.items():
        if stats['count'] > 0:
            # Add to all category
            categorized_stats['all']['sum'] += stats['sum']
            categorized_stats['all']['sum_sq'] += stats['sum_sq']
            categorized_stats['all']['sum_abs'] += stats['sum_abs']
            categorized_stats['all']['count'] += stats['count']
            
            # Check which category this parameter belongs to
            param_categorized = False
            for category, patterns in layer_patterns.items():
                for pattern in patterns:
                    if pattern in name:
                        categorized_stats[category]['sum'] += stats['sum']
                        categorized_stats[category]['sum_sq'] += stats['sum_sq']
                        categorized_stats[category]['sum_abs'] += stats['sum_abs']
                        categorized_stats[category]['count'] += stats['count']
                        param_categorized = True
                        break
                if param_categorized:
                    break
    
    # Compute final statistics
    def compute_stats_from_accumulated(stats_dict):
        """Compute statistics from accumulated values with overflow protection."""
        if stats_dict['count'] == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        count = stats_dict['count']
        
        # Compute mean
        mean = stats_dict['sum'] / count
        
        # Compute variance using numerically stable formula
        mean_sq = stats_dict['sum_sq'] / count
        variance = mean_sq - (mean ** 2)
        variance = max(0.0, variance)  # Ensure non-negative
        
        # Compute L2 norm: sqrt(sum of squares)
        l2_norm = np.sqrt(stats_dict['sum_sq'])
        
        # Compute additional distinguishing statistics
        # Standard deviation
        std = np.sqrt(variance) if variance > 0 else 0.0
        
        # Mean absolute value
        mean_abs = stats_dict['sum_abs'] / count
        
        # For skewness and kurtosis approximations
        if std > 1e-12:
            # Third moment approximation (skewness proxy)
            skewness = mean_abs / std  # Use mean absolute value as skewness proxy
            
            # Fourth moment approximation (kurtosis proxy)
            kurt = variance / (mean_abs ** 2 + 1e-12)  # Normalized variance
        else:
            skewness = 0.0
            kurt = 0.0
        
        # Apply reasonable bounds but keep distinguishing power
        mean = np.clip(mean, -1e10, 1e10)
        variance = np.clip(variance, 0, 1e10)
        l2_norm = np.clip(l2_norm, 0, 1e10)
        skewness = np.clip(skewness, 0, 1e6)
        kurt = np.clip(kurt, 0, 1e6)
        
        # Final safety check
        mean = 0.0 if (np.isinf(mean) or np.isnan(mean)) else mean
        variance = 0.0 if (np.isinf(variance) or np.isnan(variance)) else variance
        l2_norm = 0.0 if (np.isinf(l2_norm) or np.isnan(l2_norm)) else l2_norm
        skewness = 0.0 if (np.isinf(skewness) or np.isnan(skewness)) else skewness
        kurt = 0.0 if (np.isinf(kurt) or np.isnan(kurt)) else kurt
        
        return mean, variance, l2_norm, skewness, kurt
    
    # Compute statistics for each category
    stats = []
    
    # All gradients (5 stats: mean, var, L2, skew, kurt)
    all_stats = compute_stats_from_accumulated(categorized_stats['all'])
    stats.extend(all_stats)
    
    # Attention gradients (3 stats: mean, var, L2)
    attn_stats = compute_stats_from_accumulated(categorized_stats['attention'])
    stats.extend(attn_stats[:3])  # Only first 3 stats
    
    # FFN gradients (3 stats: mean, var, L2)
    ffn_stats = compute_stats_from_accumulated(categorized_stats['ffn'])
    stats.extend(ffn_stats[:3])  # Only first 3 stats
    
    # Embedding gradients (3 stats: mean, var, L2)
    emb_stats = compute_stats_from_accumulated(categorized_stats['embedding'])
    stats.extend(emb_stats[:3])  # Only first 3 stats
    
    # Add relative statistics for better discrimination
    # total_count = categorized_stats['all']['count']
    # if total_count > 0:
    #     # Relative parameter counts
    #     attn_ratio = categorized_stats['attention']['count'] / total_count
    #     ffn_ratio = categorized_stats['ffn']['count'] / total_count
    #     emb_ratio = categorized_stats['embedding']['count'] / total_count
        
    #     # Relative gradient magnitudes
    #     total_l2 = all_stats[2] if all_stats[2] > 0 else 1e-12
    #     attn_l2_ratio = attn_stats[2] / total_l2
    #     ffn_l2_ratio = ffn_stats[2] / total_l2
        
    #     stats.extend([attn_ratio, ffn_ratio, emb_ratio, attn_l2_ratio, ffn_l2_ratio])
    # else:
    #     stats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Model info (2 stats: total params, total layers) - uncomment if needed
    # total_params, total_layers = _count_model_info(model)
    # stats.extend([float(total_params), float(total_layers)])
    
    # Convert to tensor and return
    result_tensor = torch.tensor(stats, dtype=torch.float32)
    
    return result_tensor