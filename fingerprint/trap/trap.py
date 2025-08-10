import tokenize
from tqdm import tqdm
from fingerprint.fingerprint_interface import LLMFingerprintInterface
import os
import pandas as pd
import re
from fingerprint.trap.generate_prompts import generate_csv, generate_adversarial_suffix
import numpy as np
from collections import defaultdict



class TRAPFingerprint(LLMFingerprintInterface):
    """
    TRAP Fingerprint Class
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        self.n_goals = self.config.get('n_goals', 100)
        self.string_type = self.config.get('string_type', 'number')
        self.string_length = self.config.get('string_length', 3)
        self.prompt_path = self.config.get('prompt_path', None)
        self.gcg_config = self.config.get('gcg_config', {})
        # self.filtered_tokens_path = self.config.get('filtered_tokens_path', None)
        # self.filter_words_path = self.config.get('filter_words_path', "data/filter_words_number.csv")
        self.test_n_times = self.config.get('test_n_times', 5)
        self.batch_size = self.config.get('batch_size', 16)


    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        if os.path.exists(self.prompt_path) and not self.config.get('regenerate_prompts', False):
            df = pd.read_csv(self.prompt_path, dtype={'prompt': str, 'target': str, 'string_target': str})
        else:
            df = generate_csv(self.n_goals, self.string_type, self.string_length, self.prompt_path)
        self.prompts = df['prompt'].tolist()
        self.targets = df['target'].tolist()
        self.string_target = df['string_target'].tolist()


    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        # only extract fingerprint if the model is pretrained or instruct model
        if model.model_name == model.pretrained_model or model.model_name == model.instruct_model:
            torch_model, tokenizer = model.load_model()
            generated_prompts = generate_adversarial_suffix(torch_model, tokenizer, self.prompts, self.targets, self.gcg_config)
            fingerprint = generated_prompts
            return fingerprint
        else:
            return 0

    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints with batch processing optimization.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.
            batch_size (int): Number of prompts to process in each batch.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        base_fingerprint = base_model.get_fingerprint()
        batch_size = self.batch_size
        
        if not base_fingerprint or len(base_fingerprint) == 0:
            return 0.0
            
        total_matches = 0
        total_tests = 0
        
        # Create all test cases upfront for batch processing
        all_prompts = []
        all_targets = []
        all_prompt_indices = []
        
        for k, prompt in enumerate(base_fingerprint):
            target_string = self.string_target[k]
            # Repeat each prompt test_n_times
            for _ in range(self.test_n_times):
                all_prompts.append(prompt)
                all_targets.append(target_string)
                all_prompt_indices.append(k)
        
        print(f"Processing {len(all_prompts)} total tests in batches of {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_targets = all_targets[i:i+batch_size]
            batch_indices = all_prompt_indices[i:i+batch_size]
            
            try:
                # Generate responses for the entire batch
                batch_answers = testing_model.generate(batch_prompts)
                
                # Process batch results
                for j, (generated_text, target_string, prompt_idx) in enumerate(zip(batch_answers, batch_targets, batch_indices)):
                    if isinstance(generated_text, list):
                        generated_text = generated_text[0] if generated_text else ""
                    
                    if str(target_string) in generated_text:
                        total_matches += 1
                    
                    total_tests += 1
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Count failed tests
                total_tests += len(batch_prompts)
        
        # Calculate overall similarity score as the proportion of successful tests
        similarity_score = total_matches / total_tests if total_tests > 0 else 0.0
        print(f"Overall similarity score: {similarity_score:.4f} ({total_matches}/{total_tests})")
        
        return similarity_score
    