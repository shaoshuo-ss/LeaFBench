import tokenize
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
        Compare two models using their fingerprints.

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
        
        # Process prompts sequentially
        for k, prompt in enumerate(base_fingerprint):
            target_string = self.string_target[k]
            print(f"Testing prompt {k+1}/{len(base_fingerprint)} with target '{target_string}'")
            # Test each prompt multiple times
            for _ in range(self.test_n_times):
                try:
                    answer = testing_model.generate([prompt])
                    generated_text = answer[0] if isinstance(answer, list) else answer
                    print(f"Generated text: {generated_text}")
                    if str(target_string) in generated_text:
                        total_matches += 1
                    
                    total_tests += 1
                except Exception as e:
                    print(f"Error testing prompt: {e}")
                    total_tests += 1  # Still count the test even if it failed
        
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
                print(f"Generated text: {generated_text}")
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
