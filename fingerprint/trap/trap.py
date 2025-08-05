import tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading

from tqdm import tqdm
from fingerprint.fingerprint_interface import LLMFingerprintInterface
import os
import pandas as pd
import re
from fingerprint.trap.generate_prompts import generate_csv, generate_adversarial_suffix



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
        # Parallelization settings
        self.max_workers = self.config.get('max_workers', 4)  # Number of parallel workers
        self.batch_size = self.config.get('batch_size', 10)  # Batch size for prompt processing


    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        if os.path.exists(self.prompt_path) and not self.config.get('regenerate_prompts', False):
            df = pd.read_csv(self.prompt_path, dtype={'prompt': str, 'goal': str})
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

    def _test_single_prompt_multiple_times(self, testing_model, prompt, target_string, test_times):
        """
        Test a single prompt multiple times and return the number of matches.
        
        Args:
            testing_model: The model to test
            prompt: The prompt to test
            target_string: The target string to look for
            test_times: Number of times to test
            
        Returns:
            int: Number of matches found
        """
        matches = 0
        for _ in range(test_times):
            try:
                answer = testing_model.generate([prompt])
                generated_text = answer[0] if isinstance(answer, list) else answer
                if str(target_string) in generated_text:
                    matches += 1
            except Exception as e:
                # Log error but continue testing
                print(f"Error testing prompt: {e}")
                continue
        return matches

    def _test_prompt_batch(self, testing_model, prompt_batch, target_batch, test_times):
        """
        Test a batch of prompts in parallel.
        
        Args:
            testing_model: The model to test
            prompt_batch: List of prompts to test
            target_batch: List of corresponding target strings
            test_times: Number of times to test each prompt
            
        Returns:
            tuple: (total_matches, total_tests)
        """
        total_matches = 0
        total_tests = 0
        
        # Use ThreadPoolExecutor for parallel processing of multiple tests per prompt
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompt_batch))) as executor:
            futures = []
            
            for prompt, target_string in zip(prompt_batch, target_batch):
                future = executor.submit(
                    self._test_single_prompt_multiple_times,
                    testing_model, prompt, target_string, test_times
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    matches = future.result()
                    total_matches += matches
                    total_tests += test_times
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    total_tests += test_times  # Still count the tests even if they failed
        
        return total_matches, total_tests
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints with parallel processing.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        base_fingerprint = base_model.get_fingerprint()
        
        if not base_fingerprint or len(base_fingerprint) == 0:
            return 0.0
            
        total_matches = 0
        total_tests = 0
        
        # print(base_fingerprint[0])
        # Process prompts in batches for better performance
        for i in range(0, len(base_fingerprint), self.batch_size):
            batch_end = min(i + self.batch_size, len(base_fingerprint))
            prompt_batch = base_fingerprint[i:batch_end]
            target_batch = self.string_target[i:batch_end]
            
            # Process this batch
            batch_matches, batch_tests = self._test_prompt_batch(
                testing_model, prompt_batch, target_batch, self.test_n_times
            )
            
            total_matches += batch_matches
            total_tests += batch_tests
        
        # Calculate overall similarity score as the proportion of successful tests
        similarity_score = total_matches / total_tests if total_tests > 0 else 0.0
        
        return similarity_score
    
    def compare_fingerprints_original(self, base_model, testing_model):
        """
        Original implementation of compare_fingerprints for comparison.
        This method is kept for reference and fallback purposes.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        base_fingerprint = base_model.get_fingerprint()
        total_matches = 0
        total_tests = 0
        
        for k, prompt in enumerate(base_fingerprint):
            target_string = self.string_target[k]
            matches_for_this_prompt = 0
            
            for _ in range(self.test_n_times):
                answer = testing_model.generate([prompt])
                # testing_model.generate returns a list, so we take the first element
                generated_text = answer[0] if isinstance(answer, list) else answer
                
                # Check if the target string is in the generated text
                if str(target_string) in generated_text:
                    matches_for_this_prompt += 1
                    total_matches += 1
                
                total_tests += 1
            
            # Calculate success rate for this prompt
            # success_rate = matches_for_this_prompt / self.test_n_times
            # print(f"Prompt {k+1}: Target '{target_string}' found in {matches_for_this_prompt}/{self.test_n_times} tests (success rate: {success_rate:.2%})")
        
        # Calculate overall similarity score as the proportion of successful tests
        similarity_score = total_matches / total_tests if total_tests > 0 else 0.0
        # print(f"Overall similarity score: {similarity_score:.4f} ({total_matches}/{total_tests})")
        
        return similarity_score
