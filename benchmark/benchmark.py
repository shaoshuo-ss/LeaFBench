from collections import OrderedDict
import torch
import os
import pickle
from benchmark.model_pool import ModelPool
from benchmark.base_models import BaseModel
from benchmark.instruct_model import InstructModel
from benchmark.rag_models import RAGModel
from benchmark.watermark_models import WatermarkModel
from benchmark.adversarial_models import InputParaphraseModel, OutputPerturbationModel
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import ks_2samp
from scipy.spatial.distance import mahalanobis
import logging

class Benchmark:
    """ 
    Benchmark class to manage and run models for fingerprinting.
    This class initializes the model pool, loads models, and provides methods to access and run models.
    It also handles the configuration for the benchmark, including model families, pretrained models, and deploying techniques.
    """
    def __init__(self, config, accelerator=None, fingerprint_type='black-box', fingerprint_method=None):
        self.config = config
        self.accelerator = accelerator
        # Initialize the model pool with accelerator.
        # A model pool containing all base models. Any deployed model should use one of the base models in the model pool.
        self.modelpool = ModelPool(accelerator=accelerator, max_loaded_models=config.get("max_loaded_models", 1), 
                                   offload_path=config.get("offload_path", None), fingerprint_type=fingerprint_type, fingerprint_method=fingerprint_method)
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
                    if deploying_techniques.get("general_system_prompts", None) is not None:
                        # If system prompts are specified, apply them to the instruct model
                        system_prompts = deploying_techniques["general_system_prompts"]
                        self._load_system_prompt_model(
                            system_prompts, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)
                    if deploying_techniques.get("role_play_prompts", None) is not None:
                        # If role play prompts are specified, apply them to the instruct model
                        role_play_prompts = deploying_techniques["role_play_prompts"]
                        self._load_role_play_prompt_model(
                            role_play_prompts, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)
                    if deploying_techniques.get("cot_prompts", None) is not None:
                        # If COT is specified, create a COT model instance
                        cot_configs = deploying_techniques["cot_prompts"]
                        self._load_cot_model(
                            cot_configs, model_family_name, pretrained_model_name, 
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
                    if deploying_techniques.get("sampling_settings", None) is not None:
                        # If sampling settings are specified, create models with different sampling configurations
                        sampling_configs = deploying_techniques["sampling_settings"]
                        self._load_sampling_model(
                            sampling_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path)
                    if deploying_techniques.get("input_paraphrase", None) is not None:
                        # If input paraphrasing is specified, create models with different paraphrasing configurations
                        input_paraphrase_configs = deploying_techniques["input_paraphrase"]
                        self._load_input_paraphrase_model(
                            input_paraphrase_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)
                    if deploying_techniques.get("output_perturbation", None) is not None:
                        # If output perturbation is specified, create models with different perturbation configurations
                        output_perturbation_configs = deploying_techniques["output_perturbation"]
                        self._load_output_perturbation_model(
                            output_perturbation_configs, model_family_name, pretrained_model_name, 
                            instruct_model_name, instruct_model_path, default_generation_params)

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
                "type": "pretrained",
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
            self.models[instruct_model_name] = InstructModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name,
                "model_path": instruct_model_path,
                "type": "instruct",
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
            self.models[instruct_model_name + f"_with_system_prompt_{i}"] = InstructModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_with_system_prompt_{i}",
                "model_path": instruct_model_path,
                "type": "system_prompt",
                "params": {**default_generation_params, "system_prompt": system_prompt.get("template", None)}
            }, model_pool=self.modelpool, accelerator=self.accelerator)
    
    def _load_role_play_prompt_model(self, system_prompts=None, model_family_name=None, pretrained_model_name=None, 
                                  instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        """
        Load a model with a system prompt.
        """
        for i, system_prompt in enumerate(system_prompts):
            # Create a new model instance with the system prompt
            self.models[instruct_model_name + f"_with_role_play_prompt_{i}"] = InstructModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_with_role_play_prompt_{i}",
                "model_path": instruct_model_path,
                "type": "role_play_prompt",
                "params": {**default_generation_params, "role_play_prompt": system_prompt.get("template", None)}
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
            rag_config["type"] = "rag"
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
            watermark_config["type"] = "watermark"
            self.models[watermark_config["model_name"]] = WatermarkModel(watermark_config, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_cot_model(self, cot_configs=None, model_family_name=None, pretrained_model_name=None,
                        instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        for i, cot_prompt in enumerate(cot_configs):
            # Create a new model instance with the COT prompt
            self.models[instruct_model_name + f"_with_cot_prompt_{i}"] = InstructModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_with_cot_prompt_{i}",
                "type": "cot_prompt",
                "model_path": instruct_model_path,
                "params": {**default_generation_params, "cot_prompt": cot_prompt.get("template", None)}
            }, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_sampling_model(self, sampling_configs=None, model_family_name=None, pretrained_model_name=None,
                             instruct_model_name=None, instruct_model_path=None):
        """
        Load the base model with different sampling configurations.
        """
        for i, sampling_config in enumerate(sampling_configs):
            self.models[instruct_model_name + f"_sampling_{i}"] = InstructModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_sampling_{i}",
                "model_path": instruct_model_path,
                "type": "sampling",
                "params": sampling_config
            }, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_output_perturbation_model(self, output_perturbation_configs=None, model_family_name=None,
                                        pretrained_model_name=None, instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        """
        Load a model with output perturbation techniques.
        """
        for i, perturbation_config in enumerate(output_perturbation_configs):
            # Create a new model instance with the output perturbation configuration
            self.models[instruct_model_name + f"_output_perturbation_{i}"] = OutputPerturbationModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_output_perturbation_{i}",
                "model_path": instruct_model_path,
                "type": "output_perturbation",
                "params": {**default_generation_params, **perturbation_config}
            }, model_pool=self.modelpool, accelerator=self.accelerator)

    def _load_input_paraphrase_model(self, input_paraphrase_configs=None, model_family_name=None,
                                     pretrained_model_name=None, instruct_model_name=None, instruct_model_path=None, default_generation_params=None):
        """
        Load a model with input paraphrasing techniques.
        """
        for i, paraphrase_config in enumerate(input_paraphrase_configs):
            # Create a new model instance with the input paraphrasing configuration
            self.models[instruct_model_name + f"_input_paraphrase_{i}"] = InputParaphraseModel({
                "model_family": model_family_name,
                "pretrained_model": pretrained_model_name,
                "instruct_model": instruct_model_name,
                "base_model": instruct_model_name,
                "model_name": instruct_model_name + f"_input_paraphrase_{i}",
                "model_path": instruct_model_path,
                "type": "input_paraphrase",
                "params": {**default_generation_params, **paraphrase_config}
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

    def evaluate_fingerprinting_method(self, fingerprint_method, save_path):
        """
        Evaluate the fingerprinting method by comparing all models with all base models.
        
        Args:
            fingerprint_method: The fingerprint method to use for comparison
            save_path: Path to save the evaluation results
            
        Returns:
            dict: Evaluation results including metrics for each model family and type
        """
        
        logger = logging.getLogger(__name__)
        logger.info("Starting fingerprinting method evaluation...")
        
        # Get all models and separate base models (pretrained + instruct)
        all_models = self.get_all_models()
        
        # Collect base models (pretrained and instruct models)
        base_models = {}
        test_models = {}
        
        for model_name, model in all_models.items():
            if model.type in ['pretrained', 'instruct']:
                base_models[model_name] = model
            test_models[model_name] = model
        
        logger.info(f"Found {len(base_models)} base models and {len(test_models)} test models")
        
        # 1. Create similarity matrix (base models × test models)
        similarity_matrix = {}
        labels_matrix = {}  # True labels for classification
        
        base_model_names = list(base_models.keys())
        test_model_names = list(test_models.keys())
        
        for base_name in base_model_names:
            similarity_matrix[base_name] = {}
            labels_matrix[base_name] = {}
            
            for test_name in test_model_names:
                test_model = test_models[test_name]
                base_model = base_models[base_name]
                
                # Calculate similarity
                similarity = fingerprint_method.compare_fingerprints(
                    base_model=base_model, testing_model=test_model
                )
                similarity_matrix[base_name][test_name] = similarity
                
                # Determine true label (positive if same family, negative otherwise)
                # For pretrained models: positive if test model's pretrained_model matches base
                # For instruct models: positive if test model's instruct_model matches base
                is_positive = False
                if base_model.type == 'pretrained' and test_model.pretrained_model == base_name:
                    is_positive = True
                elif base_model.type == 'instruct' and test_model.instruct_model == base_name:
                    is_positive = True
                
                labels_matrix[base_name][test_name] = 1 if is_positive else 0
                
                logger.info(f"Similarity between {base_name} and {test_name}: {similarity:.4f} (label: {labels_matrix[base_name][test_name]})")
        
        # Save similarity matrix as CSV
        similarity_df = pd.DataFrame(similarity_matrix).T
        similarity_csv_path = os.path.join(save_path, "similarity_matrix.csv")
        os.makedirs(os.path.dirname(similarity_csv_path), exist_ok=True)
        similarity_df.to_csv(similarity_csv_path)
        logger.info(f"Similarity matrix saved to: {similarity_csv_path}")
        
        # 2. Calculate global optimal threshold first
        global_threshold = self._calculate_global_threshold(similarity_matrix, labels_matrix)
        logger.info(f"Global optimal threshold: {global_threshold:.4f}")
        
        # 3. Calculate metrics by model family using global threshold
        family_metrics = self._calculate_metrics_by_group(
            similarity_matrix, labels_matrix, test_models, 'model_family', global_threshold
        )
        
        # Save family metrics
        family_df = self._create_metrics_dataframe(family_metrics)
        family_csv_path = os.path.join(save_path, "metrics_by_family.csv")
        family_df.to_csv(family_csv_path)
        logger.info(f"Family metrics saved to: {family_csv_path}")
        
        # 4. Calculate metrics by model type using global threshold
        type_metrics = self._calculate_metrics_by_group(
            similarity_matrix, labels_matrix, test_models, 'type', global_threshold
        )
        
        # Save type metrics
        type_df = self._create_metrics_dataframe(type_metrics)
        type_csv_path = os.path.join(save_path, "metrics_by_type.csv")
        type_df.to_csv(type_csv_path)
        logger.info(f"Type metrics saved to: {type_csv_path}")
        
        # 5. Calculate metrics by base model using global threshold
        base_model_metrics = self._calculate_metrics_by_base_model(
            similarity_matrix, labels_matrix, base_models, global_threshold
        )
        
        # Save base model metrics
        base_model_df = self._create_base_model_metrics_dataframe(base_model_metrics)
        base_model_csv_path = os.path.join(save_path, "metrics_by_base_model.csv")
        base_model_df.to_csv(base_model_csv_path)
        logger.info(f"Base model metrics saved to: {base_model_csv_path}")
        
        # 6. Calculate overall metrics across all models using global threshold
        overall_metrics = self._calculate_overall_metrics(similarity_matrix, labels_matrix, global_threshold)
        
        # Save overall metrics
        overall_df = pd.DataFrame([overall_metrics])
        overall_csv_path = os.path.join(save_path, "overall_metrics.csv")
        overall_df.to_csv(overall_csv_path, index=False)
        logger.info(f"Overall metrics saved to: {overall_csv_path}")
        logger.info(f"Overall metrics: {overall_metrics}")
        
        logger.info("Fingerprinting method evaluation completed!")
        
        return {
            'similarity_matrix': similarity_matrix,
            'family_metrics': family_metrics,
            'type_metrics': type_metrics,
            'base_model_metrics': base_model_metrics,
            'overall_metrics': overall_metrics
        }
    
    def _create_metrics_dataframe(self, metrics_dict):
        """
        Create a DataFrame from the nested metrics dictionary.
        
        Args:
            metrics_dict: Nested dictionary with structure:
                         {group_name: {metric_type: {metric: value}}}
        
        Returns:
            pandas.DataFrame: Flattened DataFrame with hierarchical columns
        """
        
        # Flatten the nested dictionary
        flattened_data = {}
        
        for group_name, group_data in metrics_dict.items():
            for metric_type in ['pretrained', 'instruct', 'overall']:
                if metric_type in group_data:
                    for metric_name, metric_value in group_data[metric_type].items():
                        column_name = f"{metric_type}_{metric_name}"
                        if column_name not in flattened_data:
                            flattened_data[column_name] = {}
                        flattened_data[column_name][group_name] = metric_value
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data).T
        
        return df
    
    def _calculate_global_threshold(self, similarity_matrix, labels_matrix):
        """
        Calculate the global optimal threshold using all similarity scores and labels.
        
        Args:
            similarity_matrix: Dictionary of similarities {base_model: {test_model: similarity}}
            labels_matrix: Dictionary of true labels {base_model: {test_model: label}}
            
        Returns:
            float: Global optimal threshold
        """
        
        # Collect all similarities and labels
        all_similarities = []
        all_labels = []
        
        for base_name in similarity_matrix.keys():
            for test_name in similarity_matrix[base_name].keys():
                similarity = similarity_matrix[base_name][test_name]
                label = labels_matrix[base_name][test_name]
                all_similarities.append(similarity)
                all_labels.append(label)
        
        # Convert to numpy arrays
        similarities = np.array(all_similarities)
        labels = np.array(all_labels)
        
        # Calculate optimal threshold using ROC curve
        if len(np.unique(labels)) > 1:  # Need both positive and negative samples
            fpr, tpr, thresholds = roc_curve(labels, similarities)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        else:
            # If only one class present, use default threshold
            optimal_threshold = 0.5
        
        return optimal_threshold
    
    def _calculate_metrics_by_group(self, similarity_matrix, labels_matrix, test_models, group_by, global_threshold):
        """
        Calculate TP, TN, FP, FN, AUC, and accuracy metrics grouped by specified attribute.
        Separates metrics for pretrained_model, instruct_model, and overall.
        Uses a global threshold for fair comparison across groups.
        
        Args:
            similarity_matrix: Dictionary of similarities
            labels_matrix: Dictionary of true labels
            test_models: Dictionary of test models
            group_by: Attribute to group by ('model_family' or 'type')
            global_threshold: Global threshold to use for all groups
            
        Returns:
            dict: Metrics for each group, separated by base model type
        """
        
        # Group models by the specified attribute
        groups = {}
        for model_name, model in test_models.items():
            group_value = getattr(model, group_by)
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(model_name)
        
        metrics = {}
        
        for group_name, model_names in groups.items():
            # Initialize metrics for this group
            group_metrics = {
                'pretrained': {'similarities': [], 'labels': []},
                'instruct': {'similarities': [], 'labels': []},
                'overall': {'similarities': [], 'labels': []}
            }
            
            # Collect similarities and labels separated by base model type
            for base_name in similarity_matrix.keys():
                # Get base model type from the models dictionary
                base_model = None
                all_models = self.get_all_models()
                if base_name in all_models:
                    base_model = all_models[base_name]
                    base_model_type = base_model.type
                else:
                    continue
                
                for model_name in model_names:
                    if model_name in similarity_matrix[base_name]:
                        similarity = similarity_matrix[base_name][model_name]
                        label = labels_matrix[base_name][model_name]
                        
                        # Add to overall
                        group_metrics['overall']['similarities'].append(similarity)
                        group_metrics['overall']['labels'].append(label)
                        
                        # Add to specific base model type
                        if base_model_type in group_metrics:
                            group_metrics[base_model_type]['similarities'].append(similarity)
                            group_metrics[base_model_type]['labels'].append(label)
            
            # Calculate metrics for each base model type and overall
            metrics[group_name] = {}
            
            for metric_type in ['pretrained', 'instruct', 'overall']:
                similarities = np.array(group_metrics[metric_type]['similarities'])
                labels = np.array(group_metrics[metric_type]['labels'])
                
                if len(similarities) == 0:
                    # No data for this metric type
                    metrics[group_name][metric_type] = {
                        'TPR': 0.0, 'TNR': 0.0, 'FPR': 0.0, 'FNR': 0.0,
                        'AUC': 0.0, 'Accuracy': 0.0, 'Threshold': float(global_threshold),
                        'Total_Samples': 0, 'Mean_Diff': 0.0, 'TPR_at_1_FPR': 0.0,
                        'KS_Statistic': 0.0, 'Mahalanobis_Distance': 0.0
                    }
                    continue
                
                # Calculate metrics using helper function with global threshold
                calculated_metrics = self._calculate_single_metrics(similarities, labels, global_threshold)
                metrics[group_name][metric_type] = calculated_metrics
        
        return metrics
    
    def _calculate_metrics_by_base_model(self, similarity_matrix, labels_matrix, base_models, global_threshold):
        """
        Calculate metrics for each base model separately.
        This shows how well each base model can be distinguished from others.
        
        Args:
            similarity_matrix: Dictionary of similarities {base_model: {test_model: similarity}}
            labels_matrix: Dictionary of true labels {base_model: {test_model: label}}
            base_models: Dictionary of base models
            global_threshold: Global threshold to use for all groups
            
        Returns:
            dict: Metrics for each base model
        """
        
        base_model_metrics = {}
        
        for base_name in base_models.keys():
            # Get similarities and labels for this specific base model
            similarities = []
            labels = []
            
            if base_name in similarity_matrix:
                for test_name, similarity in similarity_matrix[base_name].items():
                    similarities.append(similarity)
                    labels.append(labels_matrix[base_name][test_name])
            
            # Convert to numpy arrays
            similarities = np.array(similarities)
            labels = np.array(labels)
            
            if len(similarities) == 0:
                # No data for this base model
                base_model_metrics[base_name] = {
                    'TPR': 0.0, 'TNR': 0.0, 'FPR': 0.0, 'FNR': 0.0,
                    'AUC': 0.0, 'Accuracy': 0.0, 'Threshold': float(global_threshold),
                    'Total_Samples': 0, 'Mean_Diff': 0.0, 'TPR_at_1_FPR': 0.0,
                    'KS_Statistic': 0.0, 'Mahalanobis_Distance': 0.0,
                    'Positive_Samples': 0, 'Negative_Samples': 0,
                    'Base_Model_Type': base_models[base_name].type,
                    'Model_Family': base_models[base_name].model_family
                }
                continue
            
            # Calculate metrics using helper function with global threshold
            calculated_metrics = self._calculate_single_metrics(similarities, labels, global_threshold)
            
            # Add additional information specific to base model analysis
            positive_count = np.sum(labels == 1)
            negative_count = np.sum(labels == 0)
            
            calculated_metrics.update({
                'Positive_Samples': int(positive_count),
                'Negative_Samples': int(negative_count),
                'Base_Model_Type': base_models[base_name].type,
                'Model_Family': base_models[base_name].model_family
            })
            
            base_model_metrics[base_name] = calculated_metrics
        
        return base_model_metrics
    
    def _create_base_model_metrics_dataframe(self, base_model_metrics):
        """
        Create a DataFrame from base model metrics dictionary.
        
        Args:
            base_model_metrics: Dictionary with structure:
                               {base_model_name: {metric: value}}
        
        Returns:
            pandas.DataFrame: DataFrame with base models as rows and metrics as columns
        """
        
        # Convert to DataFrame directly since it's already flat
        df = pd.DataFrame(base_model_metrics).T
        
        # Reorder columns for better readability
        preferred_order = [
            'Base_Model_Type', 'Model_Family', 'Total_Samples', 'Positive_Samples', 'Negative_Samples',
            'AUC', 'Accuracy', 'TPR', 'TNR', 'FPR', 'FNR', 
            'Mean_Diff', 'TPR_at_1_FPR', 'KS_Statistic', 'Mahalanobis_Distance', 'Threshold'
        ]
        
        # Reorder columns if they exist
        existing_columns = [col for col in preferred_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in preferred_order]
        final_order = existing_columns + remaining_columns
        
        return df[final_order]

    def _calculate_single_metrics(self, similarities, labels, threshold=None):
        """
        Calculate metrics for a single set of similarities and labels.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of true labels
            threshold: Threshold to use for predictions. If None, calculates optimal threshold.
            
        Returns:
            dict: Calculated metrics
        """
        
        if len(np.unique(labels)) > 1:  # Need both positive and negative samples
            # Calculate AUC and ROC curve
            auc = roc_auc_score(labels, similarities)
            fpr_curve, tpr_curve, thresholds_curve = roc_curve(labels, similarities)
            
            # Calculate TPR at 1% FPR
            tpr_at_1_fpr = self._calculate_tpr_at_fpr(fpr_curve, tpr_curve, target_fpr=0.01)
            
            # Calculate KS statistic
            ks_statistic = self._calculate_ks_statistic(similarities, labels)
            
            # Calculate Mahalanobis distance
            mahalanobis_distance = self._calculate_mahalanobis_distance(similarities, labels)
            
            if threshold is None:
                # Calculate optimal threshold if not provided
                optimal_idx = np.argmax(tpr_curve - fpr_curve)
                used_threshold = thresholds_curve[optimal_idx]
            else:
                # Use the provided threshold
                used_threshold = threshold
            
            # Make predictions using the threshold
            predictions = (similarities >= used_threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            
            # Calculate accuracy
            accuracy = accuracy_score(labels, predictions)
            
            # Calculate rates
            # TPR (True Positive Rate) = TP / (TP + FN) = Sensitivity = Recall
            tpr_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # TNR (True Negative Rate) = TN / (TN + FP) = Specificity
            tnr_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # FPR (False Positive Rate) = FP / (FP + TN) = 1 - TNR
            fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # FNR (False Negative Rate) = FN / (FN + TP) = 1 - TPR
            fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
        else:
            # If only one class present, set default values
            auc = 0.5
            tpr_at_1_fpr = 0.0  # Cannot calculate TPR at 1% FPR with only one class
            ks_statistic = 0.0  # Cannot calculate KS statistic with only one class
            mahalanobis_distance = 0.0  # Cannot calculate Mahalanobis distance with only one class
            used_threshold = threshold if threshold is not None else 0.5
            if len(labels) > 0 and labels[0] == 1:  # All positive
                tp, fp, tn, fn = len(labels), 0, 0, 0
                tpr_rate = 1.0  # All true positives
                tnr_rate = 0.0  # No true negatives
                fpr_rate = 0.0  # No false positives
                fnr_rate = 0.0  # No false negatives
            else:  # All negative or no labels
                tp, fp, tn, fn = 0, 0, len(labels), 0
                tpr_rate = 0.0  # No true positives
                tnr_rate = 1.0 if len(labels) > 0 else 0.0  # All true negatives
                fpr_rate = 0.0  # No false positives
                fnr_rate = 0.0  # No false negatives
            accuracy = 1.0 if len(labels) > 0 else 0.0
        
        return {
            'TPR': float(tpr_rate),
            'TNR': float(tnr_rate),
            'FPR': float(fpr_rate),
            'FNR': float(fnr_rate),
            'AUC': float(auc),
            'Accuracy': float(accuracy),
            'Threshold': float(used_threshold),
            'Total_Samples': len(similarities),
            'Mean_Diff': float(self._calculate_mean_difference(similarities, labels)),
            'TPR_at_1_FPR': float(tpr_at_1_fpr),
            'KS_Statistic': float(ks_statistic),
            'Mahalanobis_Distance': float(mahalanobis_distance)
        }

    def _calculate_overall_metrics(self, similarity_matrix, labels_matrix, global_threshold):
        """
        Calculate overall metrics across all model comparisons using global threshold.
        
        Args:
            similarity_matrix: Dictionary of similarities {base_model: {test_model: similarity}}
            labels_matrix: Dictionary of true labels {base_model: {test_model: label}}
            global_threshold: Global threshold to use for predictions
            
        Returns:
            dict: Overall metrics across all comparisons
        """
        
        # Collect all similarities and labels
        all_similarities = []
        all_labels = []
        
        for base_name in similarity_matrix.keys():
            for test_name in similarity_matrix[base_name].keys():
                similarity = similarity_matrix[base_name][test_name]
                label = labels_matrix[base_name][test_name]
                all_similarities.append(similarity)
                all_labels.append(label)
        
        # Convert to numpy arrays
        similarities = np.array(all_similarities)
        labels = np.array(all_labels)
        
        # Calculate overall metrics using the global threshold
        overall_metrics = self._calculate_single_metrics(similarities, labels, global_threshold)
        
        return overall_metrics

    def _calculate_mahalanobis_distance(self, similarities, labels):
        """
        Calculate Mahalanobis distance between positive and negative sample distributions.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of true labels
            
        Returns:
            float: Mahalanobis distance (higher means better separation)
        """
        
        # Separate similarities by true labels
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        
        if len(positive_similarities) < 2 or len(negative_similarities) < 2:
            # Cannot calculate Mahalanobis distance with insufficient samples
            return 0.0
        
        try:
            # For 1D data, we need to treat it as multivariate
            # We'll use the similarity scores directly and add some derived features
            
            # Calculate means
            pos_mean = np.mean(positive_similarities)
            neg_mean = np.mean(negative_similarities)
            
            # Create feature vectors (similarity, squared similarity, log similarity)
            # This gives us a multivariate representation for better Mahalanobis calculation
            pos_features = np.column_stack([
                positive_similarities,
                positive_similarities**2,
                np.log(np.maximum(positive_similarities, 1e-10))  # Avoid log(0)
            ])
            
            neg_features = np.column_stack([
                negative_similarities,
                negative_similarities**2,
                np.log(np.maximum(negative_similarities, 1e-10))  # Avoid log(0)
            ])
            
            # Combine all features to compute pooled covariance
            all_features = np.vstack([pos_features, neg_features])
            
            # Calculate pooled covariance matrix
            cov_matrix = np.cov(all_features.T)
            
            # Add small regularization to avoid singular matrix
            regularization = 1e-6
            cov_matrix += regularization * np.eye(cov_matrix.shape[0])
            
            # Calculate mean vectors
            pos_mean_vec = np.mean(pos_features, axis=0)
            neg_mean_vec = np.mean(neg_features, axis=0)
            
            # Calculate Mahalanobis distance between the two mean vectors
            diff_vec = pos_mean_vec - neg_mean_vec
            
            # Compute inverse of covariance matrix
            cov_inv = np.linalg.inv(cov_matrix)
            
            # Calculate Mahalanobis distance
            mahal_dist = np.sqrt(diff_vec.T @ cov_inv @ diff_vec)
            
            return mahal_dist
            
        except (np.linalg.LinAlgError, ValueError):
            # In case of numerical issues, fall back to normalized Euclidean distance
            pos_mean = np.mean(positive_similarities)
            neg_mean = np.mean(negative_similarities)
            pos_std = np.std(positive_similarities)
            neg_std = np.std(negative_similarities)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(positive_similarities) - 1) * pos_std**2 + 
                                 (len(negative_similarities) - 1) * neg_std**2) / 
                                (len(positive_similarities) + len(negative_similarities) - 2))
            
            if pooled_std > 0:
                normalized_distance = abs(pos_mean - neg_mean) / pooled_std
            else:
                normalized_distance = 0.0
                
            return normalized_distance

    def _calculate_ks_statistic(self, similarities, labels):
        """
        Calculate Kolmogorov-Smirnov statistic between positive and negative samples.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of true labels
            
        Returns:
            float: KS statistic (0-1, higher means better separation)
        """
        
        # Separate similarities by true labels
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        
        if len(positive_similarities) == 0 or len(negative_similarities) == 0:
            # Cannot calculate KS statistic with only one class
            return 0.0
        
        # Calculate KS statistic using scipy.stats.ks_2samp
        # This returns the KS statistic and p-value, we only need the statistic
        ks_statistic, _ = ks_2samp(positive_similarities, negative_similarities)
        
        return ks_statistic

    def _calculate_tpr_at_fpr(self, fpr_curve, tpr_curve, target_fpr=0.01):
        """
        Calculate TPR at a specific FPR threshold.
        
        Args:
            fpr_curve: Array of FPR values from ROC curve
            tpr_curve: Array of TPR values from ROC curve
            target_fpr: Target FPR threshold (default: 0.01 for 1%)
            
        Returns:
            float: TPR at the target FPR threshold
        """
        
        # Find the index where FPR is closest to target_fpr but not exceeding it
        # We want the highest TPR where FPR <= target_fpr
        valid_indices = np.where(fpr_curve <= target_fpr)[0]
        
        if len(valid_indices) == 0:
            # If no point has FPR <= target_fpr, return 0
            return 0.0
        
        # Among valid points, find the one with maximum TPR
        best_idx = valid_indices[np.argmax(tpr_curve[valid_indices])]
        
        return tpr_curve[best_idx]

    def _calculate_mean_difference(self, similarities, labels):
        """
        Calculate the mean difference between positive and negative predictions.
        
        Args:
            similarities: Array of similarity scores (predictions)
            labels: Array of true labels
            
        Returns:
            float: Mean of positive predictions minus mean of negative predictions
        """
        # Separate similarities by true labels
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        
        # Calculate means
        positive_mean = positive_similarities.mean() if len(positive_similarities) > 0 else 0.0
        negative_mean = negative_similarities.mean() if len(negative_similarities) > 0 else 0.0
        
        # Return the difference
        mean_diff = positive_mean - negative_mean
        
        return mean_diff

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