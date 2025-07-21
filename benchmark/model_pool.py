import logging
from collections import OrderedDict
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from accelerate import disk_offload

class ModelPool:
    def __init__(self, accelerator=None, max_loaded_models=1, offload_path=None):
        # self.models = {}  # {model_name: model_instance}
        self.tokenizers = OrderedDict()  # {model_name: tokenizer_instance}
        self.model_paths = OrderedDict()  # {model_name: model_path}
        self.accelerator = accelerator
        self.current_loaded_models = OrderedDict()  # {model_name: model_instance}
        self.max_loaded_models = max_loaded_models
        self.offload_path = offload_path
        if self.offload_path:
            os.makedirs(self.offload_path, exist_ok=True)

    def register_model(self, model_name, model_path):
        """
        Register the model path, but do not load the model.
        """
        self.model_paths[model_name] = model_path
        # with init_empty_weights():
        # self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_path) if model_path else None
        # tokenizer = AutoTokenizer.from_pretrained(model_path) if model_path else None
        # Set pad token if it doesn't exist
        # if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token

        # self.tokenizers[model_name] = tokenizer

    def get_tokenizer(self, model_name):
        """
        Get the tokenizer for the specified model, load it on demand and cache it.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_paths[model_name]) if self.model_paths[model_name] else None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_model(self, model_name, type=None):
        """
        Get the model object, load it to the specified device (default GPU) on demand.
        Uses accelerator for proper multi-GPU management if available.
        """
        logger = logging.getLogger(__name__)
        if self.model_paths[model_name] is None:
            raise ValueError(f"Model {model_name} not registered.")
        else:
            if model_name not in self.current_loaded_models.keys():
                if len(self.current_loaded_models) >= self.max_loaded_models:
                    # Unload the least recently used model
                    oldest_model_name = next(iter(self.current_loaded_models))
                    logger.info(f"Unloading {oldest_model_name} to make room for {model_name}")
                    
                    # Get the model to be unloaded
                    model_to_unload = self.current_loaded_models[oldest_model_name]
                    
                    # Completely unload the model from memory
                    self._completely_unload_model(model_to_unload, oldest_model_name, logger)
                    
                    # Remove from the loaded models dictionary
                    del self.current_loaded_models[oldest_model_name]
                    
                    logger.info(f"Successfully unloaded {oldest_model_name} and freed all memory")
        
                # Load the new model
                if type == "adapter":
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        self.model_paths[model_name], 
                        device_map="balanced", 
                        torch_dtype=torch.float16
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_paths[model_name], 
                        device_map="balanced", 
                        torch_dtype=torch.float16
                    )
                self.current_loaded_models[model_name] = model
                logger.info(f"{len(self.current_loaded_models)} models loaded in the pool")
                
            else:
                # Move to end (most recently used)
                model = self.current_loaded_models.pop(model_name)
                self.current_loaded_models[model_name] = model
                
            return self.current_loaded_models[model_name]

    def list_models(self):
        """
        List all registered models.
        """
        return list(self.model_paths.keys())
    
    def _completely_unload_model(self, model, model_name, logger):
        """
        Completely unload a model from memory, including GPU and CPU memory.
        """
        import gc
        
        try:
            # Step 1: Clear all hooks that might keep references
            if hasattr(model, '_forward_hooks'):
                model._forward_hooks.clear()
            if hasattr(model, '_backward_hooks'):
                model._backward_hooks.clear()
            if hasattr(model, '_forward_pre_hooks'):
                model._forward_pre_hooks.clear()
            
            # Step 2: Handle models with device_map (distributed across devices)
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Model {model_name} has device_map, performing distributed cleanup")
                
                # For models with device_map, we need to clear each device
                for module_name, device in model.hf_device_map.items():
                    try:
                        module = model
                        for attr in module_name.split('.'):
                            if attr:
                                module = getattr(module, attr)
                        
                        # Move module to CPU and clear its data
                        if hasattr(module, 'weight') and module.weight is not None:
                            module.weight.data = module.weight.data.cpu()
                            del module.weight
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data.cpu()
                            del module.bias
                    except Exception as e:
                        logger.warning(f"Error clearing module {module_name}: {e}")
                
                # Clear the device map
                if hasattr(model, 'hf_device_map'):
                    del model.hf_device_map
            
            # Step 3: Move all parameters and buffers to CPU and delete them
            params_to_delete = []
            buffers_to_delete = []
            
            for name, param in model.named_parameters():
                if param.device.type == 'cuda':
                    param.data = param.data.cpu()
                params_to_delete.append((name, param))
            
            for name, buffer in model.named_buffers():
                if buffer.device.type == 'cuda':
                    buffer.data = buffer.data.cpu()
                buffers_to_delete.append((name, buffer))
            
            # Clear parameter and buffer references
            for name, param in params_to_delete:
                try:
                    delattr(model, name.split('.')[-1]) if '.' not in name else None
                except:
                    pass
                del param
                
            for name, buffer in buffers_to_delete:
                try:
                    delattr(model, name.split('.')[-1]) if '.' not in name else None
                except:
                    pass
                del buffer
            
            # Step 4: Clear model's internal state
            if hasattr(model, 'config'):
                del model.config
            if hasattr(model, 'generation_config'):
                del model.generation_config
            
            # Step 5: Force model to CPU (if not already done)
            try:
                model.cpu()
            except:
                pass
            
            # Step 6: Multiple rounds of cleanup
            for i in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"Model {model_name} completely unloaded from memory")
            
        except Exception as e:
            logger.error(f"Error during complete model unloading: {e}")
            # Fallback: still try basic cleanup
            try:
                model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass