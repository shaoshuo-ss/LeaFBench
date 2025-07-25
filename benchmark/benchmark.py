from collections import OrderedDict
import torch
import os
import pickle
from benchmark.model_pool import ModelPool
from benchmark.base_models import BaseModel
from benchmark.rag_models import RAGModel
from benchmark.watermark_models import WatermarkModel
import logging

class Benchmark:
    """ 
    Benchmark class to manage and run models for fingerprinting.
    This class initializes the model pool, loads models, and provides methods to access and run models.
    It also handles the configuration for the benchmark, including model families, pretrained models, and deploying techniques.
    """
    def __init__(self, config, accelerator=None, fingerprint_type='black-box'):
        self.config = config
        self.accelerator = accelerator
        # Initialize the model pool with accelerator.
        # A model pool containing all base models. Any deployed model should use one of the base models in the model pool.
        self.modelpool = ModelPool(accelerator=accelerator, max_loaded_models=config.get("max_loaded_models", 1), 
                                   offload_path=config.get("offload_path", None), fingerprint_type=fingerprint_type)
        self.models = OrderedDict()
        default_generation_params = config.get("default_generation_params", {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "max_input_length": 512
        })
        # register all the base models
        for model_family in self.config['models']:
            model_family_name = model_family.get("model_family", None)
            
            # load the pretrained model
            pretrained_model = model_family.get("pretrained_model", None)
            pretrained_model_name, pretrained_model_path = self._load_pretrained_model(
                pretrained_model, model_family_name, default_generation_params)
            
            # load the instruct model
            instruct_model = model_family.get("instruct_model", None)
            instruct_model_name, instruct_model_path = self._load_instruct_model(
                instruct_model, model_family_name, pretrained_model_name, default_generation_params)

            # apply deploying techniques to instruct model and construct new model instances
            # white-box fingerprinting methods do not require testing deploying techniques, so we skip this step
            if fingerprint_type == 'black-box':
                deploying_techniques = self.config.get("deploying_techniques", {})
                if deploying_techniques:
                    if deploying_techniques.get("system_prompts", None) is not None:
                        # If system prompts are specified, apply them to the instruct model
                        system_prompts = deploying_techniques["system_prompts"]
                        self._load_system_prompt_model(
                            system_prompts, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)
                    if deploying_techniques.get("rag", None) is not None:
                        # If RAG is specified, create a RAG model instance
                        rag_configs = deploying_techniques["rag"]
                        self._load_rag_model(
                            rag_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, default_generation_params)
                    if deploying_techniques.get("watermark", None) is not None:
                        # If watermarking is specified, create a Watermark model instance
                        watermark_configs = deploying_techniques["watermark"]
                        self._load_watermark_model(
                            watermark_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, default_generation_params)
                    if deploying_techniques.get("cot", None) is not None:
                        # If COT is specified, create a COT model instance
                        cot_configs = deploying_techniques["cot"]
                        self._load_cot_model(
                            cot_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)
                    if deploying_techniques.get("sampling_settings", None) is not None:
                        # If sampling settings are specified, create models with different sampling configurations
                        sampling_configs = deploying_techniques["sampling_settings"]
                        self._load_sampling_model(
                            sampling_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path)

            # load the predeployed models
            predeployed_models = model_family.get("predeployed_models", [])
            self._load_predeployed_model(
                predeployed_models, model_family_name, 
                pretrained_model_name, instruct_model_name, default_generation_params)
            
        
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

    def _load_pretrained_model(self, pretrained_model=None, model_family_name=None, default_generation_params=None):
        """
        Load a pretrained model from the model pool.
        """
        if pretrained_model is None:
            raise ValueError("Pretrained model name must be specified.")
        pretrained_model_name = pretrained_model.get("model_name", None)
        pretrained_model_path = pretrained_model.get("model_path", None)
        if pretrained_model_name is not None and pretrained_model_path is not None:
            # Register the model in the model pool
            self.modelpool.register_model(pretrained_model_name, pretrained_model_path)
            # Create a model instance and store it in the models dictionary
            self.models[pretrained_model_name] = BaseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": None,
                "base_model": pretrained_model_name,
                "model_name": pretrained_model_name,
                "model_path": pretrained_model_path,
                "params": default_generation_params
            }, model_pool=self.modelpool, accelerator=self.accelerator)
        return pretrained_model_name, pretrained_model_path
    
    def _load_instruct_model(self, instruct_model=None, model_family_name=None, pretrained_model_name=None, default_generation_params=None):
        """
        Load an instruct model from the model pool.
        """
        if instruct_model is None:
            raise ValueError("Instruct model name must be specified.")
        instruct_model_name = instruct_model.get("model_name", None)
        instruct_model_path = instruct_model.get("model_path", None)
        if instruct_model_name is not None and instruct_model_path is not None:
            # Register the model in the model pool
            self.modelpool.register_model(instruct_model_name, instruct_model_path)
            # Create a model instance and store it in the models dictionary
            self.models[instruct_model_name] = BaseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name,
                "model_path": instruct_model_path,
                "params": default_generation_params
            }, model_pool=self.modelpool, accelerator=self.accelerator)
        return instruct_model_name, instruct_model_path
    
    def _load_system_prompt_model(self, system_prompts=None, model_family_name=None, pretrained_model_name=None, 
                                  instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        """
        Load a model with a system prompt.
        """
        for i, system_prompt in enumerate(system_prompts):
            # Create a new model instance with the system prompt
            self.models[instruct_model_name + f"_with_system_prompt_{i}"] = BaseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_with_system_prompt_{i}",
                "model_path": instruct_model_path,
                "params": {**default_generation_params, "system_prompt": system_prompt.get("template", None)}
            }, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_rag_model(self, rag_configs=None, model_family_name=None, pretrained_model_name=None,
                        instruct_model_name=None, default_generation_params=None):
        """
        Load a RAG model with the specified configuration.
        """
        for i, rag_config in enumerate(rag_configs):
            rag_config["model_family"] = model_family_name
            rag_config["pretrained_model"] = pretrained_model_name
            rag_config["instruct_model"] = instruct_model_name
            rag_config["base_model"] = instruct_model_name
            rag_config["model_name"] = instruct_model_name + "_rag_" + str(i)
            rag_config["params"] = default_generation_params
            self.models[rag_config["model_name"]] = RAGModel(rag_config, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_watermark_model(self, watermark_configs=None, model_family_name=None, pretrained_model_name=None,
                              instruct_model_name=None, default_generation_params=None):
        """
        Load a watermark model with the specified configuration.
        """
        for i, watermark_config in enumerate(watermark_configs):
            watermark_config["model_family"] = model_family_name
            watermark_config["pretrained_model"] = pretrained_model_name
            watermark_config["instruct_model"] = instruct_model_name
            watermark_config["base_model"] = instruct_model_name
            watermark_config["model_name"] = instruct_model_name + "_watermark_" + str(i)
            watermark_config["params"] = default_generation_params
            self.models[watermark_config["model_name"]] = WatermarkModel(watermark_config, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_cot_model(self, cot_configs=None, model_family_name=None, pretrained_model_name=None,
                        instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        for i, cot_prompt in enumerate(cot_configs):
            # Create a new model instance with the COT prompt
            self.models[instruct_model_name + f"_with_cot_prompt_{i}"] = BaseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_with_cot_prompt_{i}",
                "model_path": instruct_model_path,
                "params": {**default_generation_params, "cot_prompt": cot_prompt.get("template", None)}
            }, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_sampling_model(self, sampling_configs=None, model_family_name=None, pretrained_model_name=None,
                             instruct_model_name=None, instruct_model_path=None):
        """
        Load the base model with different sampling configurations.
        """
        for i, sampling_config in enumerate(sampling_configs):
            self.models[instruct_model_name + f"_sampling_{i}"] = BaseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_sampling_{i}",
                "model_path": instruct_model_path,
                "params": sampling_config
            }, model_pool=self.modelpool, accelerator=self.accelerator)
    
    def _load_predeployed_model(self, predeployed_models=None, model_family_name=None,
                                pretrained_model_name=None, instruct_model_name=None, default_generation_params=None):
        """
        Load predeployed models.
        """
        for predeployed_model in predeployed_models:
            predeployed_model_name = predeployed_model.get("model_name", None)
            predeployed_model_path = predeployed_model.get("model_path", None)
            predeployed_model_type = predeployed_model.get("type", None)
            if predeployed_model_name is not None and predeployed_model_path is not None:
                # Register the model in the model pool
                self.modelpool.register_model(predeployed_model_name, predeployed_model_path)
                # Create a model instance and store it in the models dictionary
                self.models[predeployed_model_name] = BaseModel({
                    "model_family": model_family_name,
                    "pretrained_model": pretrained_model_name,
                    "instruct_model": instruct_model_name,
                    "base_model": predeployed_model_name,
                    "model_name": predeployed_model_name,
                    "model_path": predeployed_model_path,
                    "type": predeployed_model_type,
                    "params": default_generation_params
                }, model_pool=self.modelpool, accelerator=self.accelerator)

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
    
    def get_all_models(self):
        """
        Get all models, both training and testing.
        """
        return self.models


def load_fingerprints(cached_fingerprints_path, benchmark):
    """
    Resume cached fingerprints if available.
    """
    logger = logging.getLogger(__name__)
    if cached_fingerprints_path and os.path.exists(cached_fingerprints_path):
        logger.info(f"Loading cached fingerprints from {cached_fingerprints_path}")
        
        # Check file size first (empty files will cause corruption errors)
        file_size = os.path.getsize(cached_fingerprints_path)
        if file_size == 0:
            logger.warning(f"Cached fingerprints file {cached_fingerprints_path} is empty. Starting fresh.")
            return
        
        # Try to load the cached fingerprints with error handling
        try:
            cached_fingerprints = torch.load(cached_fingerprints_path, map_location='cpu')
            
            # Validate the loaded data
            if not isinstance(cached_fingerprints, dict):
                logger.warning(f"Invalid cached fingerprints format. Expected dict, got {type(cached_fingerprints)}. Starting fresh.")
                return
            
            # Load the cached fingerprints
            benchmark_models = benchmark.get_all_models()
            loaded_count = 0
            for model_name in cached_fingerprints.keys():
                if model_name in benchmark_models:
                    benchmark_models[model_name].set_fingerprint(cached_fingerprints[model_name])
                    loaded_count += 1
                    
            logger.info(f"Successfully loaded {loaded_count} cached fingerprints")
            
        except (RuntimeError, pickle.UnpicklingError, EOFError, OSError) as e:
            logger.error(f"Error loading cached fingerprints: {e}")
            logger.warning(f"Cached fingerprints file {cached_fingerprints_path} appears to be corrupted. Starting fresh.")
            
            # Optionally, backup the corrupted file and remove it
            corrupted_backup = cached_fingerprints_path + ".corrupted"
            try:
                os.rename(cached_fingerprints_path, corrupted_backup)
                logger.info(f"Corrupted file backed up to {corrupted_backup}")
            except Exception as backup_error:
                logger.warning(f"Could not backup corrupted file: {backup_error}")
                try:
                    os.remove(cached_fingerprints_path)
                    logger.info(f"Removed corrupted file {cached_fingerprints_path}")
                except Exception as remove_error:
                    logger.warning(f"Could not remove corrupted file: {remove_error}")
    else:
        logger.info(f"No cached fingerprints found at {cached_fingerprints_path}. Starting fresh.")


def save_fingerprints(cached_fingerprints_path, benchmark):
    """
    Save fingerprints to the specified path with robust error handling.
    """
    logger = logging.getLogger(__name__)
    if cached_fingerprints_path:
        logger.info(f"Saving fingerprints to {cached_fingerprints_path}")
        os.makedirs(os.path.dirname(cached_fingerprints_path), exist_ok=True)
        
        fingerprints = {model_name: model.get_fingerprint() 
                        for model_name, model in benchmark.get_all_models().items() 
                        if model.get_fingerprint() is not None}
            
        torch.save(fingerprints, cached_fingerprints_path)
    else:
        logger.warning("No path specified for saving fingerprints. Skipping save operation.")