from collections import OrderedDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmark.benchmark_interface import BaseModel
import logging

class ModelPool:
    def __init__(self, accelerator=None, max_loaded_models=1, offload_path=None):
        # self.models = {}  # {model_name: model_instance}
        self.tokenizers = OrderedDict()  # {model_name: tokenizer_instance}
        self.model_paths = OrderedDict()  # {model_name: model_path}
        self.accelerator = accelerator
        self.current_loaded_models = OrderedDict()  # {model_name: model_instance}
        self.max_loaded_models = max_loaded_models
        self.offload_path = offload_path

    def register_model(self, model_name, model_path):
        """
        Register the model path, but do not load the model.
        """
        self.model_paths[model_name] = model_path
        # with init_empty_weights():
        # self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_path) if model_path else None
        tokenizer = AutoTokenizer.from_pretrained(model_path) if model_path else None
        # Set pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizers[model_name] = tokenizer

    def get_tokenizer(self, model_name):
        """
        Get the tokenizer for the specified model, load it on demand and cache it.
        """
        if self.tokenizers[model_name] is None:
            raise ValueError(f"Model {model_name} not registered.")
        else:
            return self.tokenizers[model_name]

    def get_model(self, model_name):
        """
        Get the model object, load it to the specified device (default GPU) on demand.
        Uses accelerator for proper multi-GPU management if available.
        """
        logger = logging.getLogger(__name__)
        if self.model_paths[model_name] is None:
            raise ValueError(f"Model {model_name} not registered.")
        else:
            if model_name not in self.current_loaded_models.keys():
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_paths[model_name], 
                    device_map="balanced", 
                    torch_dtype=torch.float16
                )
                
                if len(self.current_loaded_models) > 0:
                    # Unload the least recently used model
                    oldest_model_name = next(iter(self.current_loaded_models))
                    logger.info(f"Unloading {oldest_model_name} to make room for {model_name}")
                    del self.current_loaded_models[oldest_model_name]
                    torch.cuda.empty_cache()
                
                # Load the new model
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

class Benchmark:
    def __init__(self, config, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        # Initialize the model pool with accelerator.
        # A model pool containing all base models. Any deployed model should use one of the base models in the model pool.
        self.modelpool = ModelPool(accelerator=accelerator, max_loaded_models=config.get("max_loaded_models", 1))
        self.models = OrderedDict()
        # register all the base models
        for model_family in self.config['models']:
            model_family_name = model_family.get("model_family", None)
            for pretrained_model in model_family.get("pretrained_models", []):
                pretrained_model_name = pretrained_model.get("model_name", None)
                pretrained_model_path = pretrained_model.get("model_path", None)
                if pretrained_model_name is not None and pretrained_model_path is not None:
                    # Register the model in the model pool
                    self.modelpool.register_model(pretrained_model_name, pretrained_model_path)
                    # Create a model instance and store it in the models dictionary
                    self.models[pretrained_model_name] = BaseModel({
                        "model_family": model_family_name,
                        "pretrained_model": pretrained_model_name,
                        "base_model": pretrained_model_name,
                        "model_name": pretrained_model_name,
                        "model_path": pretrained_model_path,
                        "params": pretrained_model.get("params", {})
                    }, model_pool=self.modelpool, accelerator=accelerator)
                for predeployed_model in pretrained_model.get("predeployed_models", []):
                    predeployed_model_name = predeployed_model.get("model_name", None)
                    predeployed_model_path = predeployed_model.get("model_path", None)
                    if predeployed_model_name is not None and predeployed_model_path is not None:
                        # Register the model in the model pool
                        self.modelpool.register_model(predeployed_model_name, predeployed_model_path)
                        # Create a model instance and store it in the models dictionary
                        self.models[predeployed_model_name] = BaseModel({
                            "model_family": model_family_name,
                            "pretrained_model": pretrained_model_name,
                            "base_model": predeployed_model_name,
                            "model_name": predeployed_model_name,
                            "model_path": predeployed_model_path,
                            "params": predeployed_model.get("params", {})
                        }, model_pool=self.modelpool, accelerator=accelerator)
        # TODO: register models with deployed techniques.


        # prepare all models in the model pool
        # self.modelpool.prepare_models()
        
        # Split training models and testing models.
        self.training_models = {}
        for model_name in self.config["training_data"]["models"]:
            if model_name in self.models.keys():
                self.training_models[model_name] = self.models[model_name]

        # Construct testing models set (models not in training set)
        self.testing_models = {}
        training_model_names = set(self.training_models.keys())
        for model_name, model_instance in self.models.items():
            if model_name not in training_model_names:
                self.testing_models[model_name] = model_instance

    def get_model_pool(self):
        """
        Get the model pool instance.
        """
        return self.modelpool
    
    def get_model(self, model_name):
        """
        Get a specific model instance by name.
        """
        return self.models.get(model_name)
    
    def get_training_models(self):
        """
        Get all training models.
        """
        return self.training_models
    
    def get_testing_models(self):
        """
        Get all testing models.
        """
        return self.testing_models
    
    def list_available_models(self):
        """
        List all available model names.
        """
        return list(self.models.keys())