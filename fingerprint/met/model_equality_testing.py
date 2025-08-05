from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.met.wikipedia_sampler import sample_wikipedia_texts
from fingerprint.met.mmd_utils import met_similarity_test
import torch
import torch.nn.functional as F
import os
import pandas as pd
import logging

class METFingerprint(LLMFingerprintInterface):
    """
    MET fingerprinting method.
    This class implements the MET fingerprinting technique.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config, accelerator)
        self.n_samples = self.config.get('n_samples', 25)
        self.samples_path = self.config.get('samples_path', None)
        self.regenerate_samples = self.config.get('regenerate_samples', False)
        self.text_length = self.config.get('text_length', 100)
        self.response_length = self.config.get('response_length', 50)
        # Batch processing settings
        self.batch_size = self.config.get('batch_size', 5)
        # Multiple runs settings
        self.n_runs = self.config.get('n_runs', 100)
        # MMD test settings
        self.gamma = self.config.get('gamma', 1.0)
        self.n_bootstrap = self.config.get('n_bootstrap', 1000)
        self.alpha = self.config.get('alpha', 0.05)
        self.random_seed = self.config.get('random_seed', 42)
    
    @property
    def logger(self):
        """Get logger instance for this class."""
        return logging.getLogger(__name__)
        

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods by sampling Wikipedia texts.
        
        Args:
            train_models (optional): Models to train, if necessary.
        """
        # Check if samples already exist and we don't need to regenerate
        if (self.samples_path and 
            os.path.exists(self.samples_path) and 
            not self.regenerate_samples):
            self.logger.info(f"Loading existing samples from {self.samples_path}")
            df = pd.read_csv(self.samples_path)
            self.samples = df.to_dict('records')
            self.prompts = df['prompt'].tolist()
            self.text_samples = df['text_sample'].tolist()
        else:
            self.logger.info("Generating new Wikipedia samples...")
            self.samples = sample_wikipedia_texts(
                n_samples=self.n_samples,
                text_length=self.text_length,
                cache_path=self.samples_path,
                regenerate=self.regenerate_samples
            )
            self.prompts = [sample['prompt'] for sample in self.samples]
            self.text_samples = [sample['text_sample'] for sample in self.samples]
        
        self.logger.info(f"Prepared {len(self.samples)} samples for MET fingerprinting")

    def _generate_single_batch(self, model, batch_prompts, batch_start):
        """
        Generate responses for a single batch of prompts.
        
        Args:
            model: The model to use for generation
            batch_prompts: List of prompts in this batch
            batch_start: Starting index of this batch
            
        Returns:
            list: List of truncated responses for the batch
        """
        batch_fingerprint = []
        
        try:
            # Try batch generation first
            batch_responses = model.generate(batch_prompts)
            
            # Process each response in the batch
            for i, response in enumerate(batch_responses):
                try:
                    # Extract the actual text from response
                    if isinstance(response, list) and len(response) > 0:
                        response_text = response[0]
                    else:
                        response_text = str(response)
                    
                    # Truncate the response to the specified length
                    truncated_response = response_text[:self.response_length]
                    batch_fingerprint.append(truncated_response)
                    
                except Exception as e:
                    print(f"Error processing response {batch_start + i + 1}: {e}")
                    batch_fingerprint.append("")
                    
        except Exception as e:
            print(f"Batch generation failed for batch starting at {batch_start}: {e}")
            print("Falling back to individual prompt processing for this batch...")
            
            # Fallback to individual processing for this batch
            for i, prompt in enumerate(batch_prompts):
                try:
                    response = model.generate([prompt])
                    
                    if isinstance(response, list) and len(response) > 0:
                        response_text = response[0]
                    else:
                        response_text = str(response)
                    
                    truncated_response = response_text[:self.response_length]
                    batch_fingerprint.append(truncated_response)
                    
                except Exception as e:
                    print(f"Error generating individual response {batch_start + i + 1}: {e}")
                    batch_fingerprint.append("")
        
        return batch_fingerprint

    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model using batch processing and multiple runs.

        Args:
            model: The model to extract the fingerprint.
        
        Returns:
            list: List of truncated responses (fingerprint) from multiple runs.
        """
        if not hasattr(self, 'prompts') or not self.prompts:
            raise ValueError("No prompts available. Please run prepare() first.")
        
        self.logger.info(f"Generating fingerprint for model using {len(self.prompts)} prompts, "
                   f"batch size {self.batch_size}, {self.n_runs} runs")
        
        all_fingerprints = []
        
        # Perform multiple runs
        for run_idx in range(self.n_runs):
            self.logger.info(f"Starting run {run_idx + 1}/{self.n_runs}")
            
            fingerprint = []
            
            # Process prompts in batches
            for batch_start in range(0, len(self.prompts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.prompts))
                batch_prompts = self.prompts[batch_start:batch_end]
                
                # Generate responses for this batch
                batch_fingerprint = self._generate_single_batch(model, batch_prompts, batch_start)
                fingerprint.extend(batch_fingerprint)
                
                # Progress tracking (less verbose for multiple runs)
                if run_idx == 0:  # Only show detailed progress for first run
                    batch_num = batch_start // self.batch_size + 1
                    total_batches = (len(self.prompts) + self.batch_size - 1) // self.batch_size
                    print(f"Processed batch {batch_num}/{total_batches} ({len(fingerprint)}/{len(self.prompts)} prompts)")
            
            all_fingerprints.extend(fingerprint)
            
            if (run_idx + 1) % 10 == 0:
                self.logger.info(f"Completed {run_idx + 1}/{self.n_runs} runs")
        
        self.logger.info(f"Fingerprint generation completed. Generated {len(all_fingerprints)} total responses "
                   f"({self.n_runs} runs × {len(self.prompts)} prompts)")
        
        if all_fingerprints:
            avg_length = sum(len(resp) for resp in all_fingerprints) / len(all_fingerprints)
            self.logger.info(f"Average response length: {avg_length:.1f} characters")
        
        return all_fingerprints
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two fingerprints using Maximum Mean Discrepancy (MMD) with Hamming kernel.
        
        Performs a statistical hypothesis test where:
        - H0: Both models generate responses from the same distribution
        - H1: Models generate responses from different distributions
        
        Uses bootstrap sampling from base model to construct null distribution,
        then computes p-value for the observed MMD statistic.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.
        
        Returns:
            float: Similarity score based on p-value (higher = more similar).
        """
        # Get fingerprints from both models
        base_fingerprint = base_model.get_fingerprint()
        test_fingerprint = testing_model.get_fingerprint()
        
        # Validate fingerprints
        if not base_fingerprint or not test_fingerprint:
            self.logger.warning("One or both fingerprints are empty")
            return 0.0
        
        if len(base_fingerprint) != len(test_fingerprint):
            self.logger.warning(f"Fingerprint lengths differ: base={len(base_fingerprint)}, test={len(test_fingerprint)}")
            # Truncate to shorter length
            min_length = min(len(base_fingerprint), len(test_fingerprint))
            base_fingerprint = base_fingerprint[:min_length]
            test_fingerprint = test_fingerprint[:min_length]
        
        self.logger.info(f"Comparing fingerprints using MMD test...")
        self.logger.info(f"Base fingerprint length: {len(base_fingerprint)}")
        self.logger.info(f"Test fingerprint length: {len(test_fingerprint)}")
        self.logger.info(f"MMD parameters: gamma={self.gamma}, n_bootstrap={self.n_bootstrap}, alpha={self.alpha}")
        
        # Perform MMD similarity test
        similarity_score, test_stats = met_similarity_test(
            base_fingerprints=base_fingerprint,
            test_fingerprints=test_fingerprint,
            gamma=self.gamma,
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            random_seed=self.random_seed
        )
        
        # Log detailed results
        self.logger.info(f"MMD Test Results:")
        self.logger.info(f"  Observed MMD: {test_stats['observed_mmd']:.6f}")
        self.logger.info(f"  P-value: {test_stats['p_value']:.6f}")
        self.logger.info(f"  Null MMD mean: {test_stats['null_mmds_mean']:.6f}")
        self.logger.info(f"  Null MMD std: {test_stats['null_mmds_std']:.6f}")
        self.logger.info(f"  Models are {'SIMILAR' if test_stats['is_similar'] else 'DIFFERENT'} (α={self.alpha})")
        self.logger.info(f"  P-value similarity: {test_stats['p_value_similarity']:.6f}")
        self.logger.info(f"  MMD similarity: {test_stats['mmd_similarity']:.6f}")
        self.logger.info(f"  Percentile similarity: {test_stats['percentile_similarity']:.6f}")
        
        # Store test statistics for potential later use
        self.last_test_stats = test_stats
        
        return similarity_score