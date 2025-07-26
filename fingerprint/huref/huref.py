import torch
import torch.nn.functional as F
import os
import numpy as np
from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.huref.invariant_terms import MeanPooling, CNNEncode, get_invariant_terms
from fingerprint.huref.sort_tokens_frequency import sort_tokens_frequency

class HuRefFingerprint(LLMFingerprintInterface):
    """
    HuRef fingerprinting method.
    This class implements the HuRef fingerprinting technique.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config, accelerator)

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods.
        """
        # Implementation for preparing HuRef fingerprinting
        self.feature_extraction_method = self.config.get('feature_extraction_method', 'MeanPooling')
        if self.feature_extraction_method == "MeanPooling":
            self.feature_extractor = MeanPooling()
        elif self.feature_extraction_method == "CNN":
            self.feature_extractor = CNNEncode(self.config.get('encoder_path', None))
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.feature_extraction_method}")


    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model.

        Args:
            model: The model to extract the fingerprint.
        
        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        # Step 1: Get model's state_dict and selected_tokens
        torch_model, tokenizer = model.load_model()
        state_dict = torch_model.state_dict()
        
        # Get model name
        model_name = model.pretrained_model
        
        # Get sorted tokens
        selected_tokens = self._get_selected_tokens(tokenizer, model_name)
        
        # Step 2: Get invariant terms using get_invariant_terms function
        invariant_terms = get_invariant_terms(state_dict, model_name, selected_tokens)
        
        # Step 3: Use feature_extractor to get unified feature
        if self.accelerator is not None:
            invariant_terms = invariant_terms.to(self.accelerator.device)
        
        self.feature_extractor = self.feature_extractor.to(self.accelerator.device) if self.accelerator else self.feature_extractor
        feature_vector = self.feature_extractor(invariant_terms)
        
        return feature_vector
    
    def _get_selected_tokens(self, tokenizer, name):
        """
        Get selected tokens, either from existing file or by computing and saving them.
        
        Args:
            model: The model to get tokens for
            name: Model name for file naming
            
        Returns:
            list: Selected tokens
        """
        # Define paths (these will be set from config or default values)
        sorted_tokens_path = self.config.get('sorted_tokens_path', './data/sorted_tokens/')
        num_tokens = self.config.get('num_tokens', 4096)
        datanum = self.config.get('datanum', 400000)
        num_processes = self.config.get('num_processes', 40)
        
        # Ensure directory exists
        os.makedirs(sorted_tokens_path, exist_ok=True)
        
        # Check if sorted tokens file exists
        tokens_file_path = os.path.join(sorted_tokens_path, f"{name}.txt")
        
        if os.path.exists(tokens_file_path):
            # Read existing sorted tokens file using numpy for better efficiency
            try:
                sorted_tokens = np.loadtxt(tokens_file_path, dtype=int).tolist()
            except:
                # Fallback to line-by-line reading if numpy fails
                sorted_tokens = []
                with open(tokens_file_path, 'r') as file:
                    for line in file:
                        sorted_tokens.append(int(line.strip()))
        else:
            # Compute and save sorted tokens
            sort_tokens_frequency(
                tokenizer=tokenizer,
                model_name=name,
                savepath=sorted_tokens_path,
                datanum=datanum,
                num_processes=num_processes
            )
            
            # Read the newly created file using numpy for better efficiency
            try:
                sorted_tokens = np.loadtxt(tokens_file_path, dtype=int).tolist()
            except:
                # Fallback to line-by-line reading if numpy fails
                sorted_tokens = []
                with open(tokens_file_path, 'r') as file:
                    for line in file:
                        sorted_tokens.append(int(line.strip()))
        
        # Select the last num_tokens tokens
        selected_tokens = sorted_tokens[-num_tokens:]
        
        return selected_tokens
        
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two fingerprints using cosine similarity.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.
        
        Returns:
            float: Cosine similarity score between the two fingerprints (scaling to [0,1]).
        """
        # Ensure both fingerprints are tensors
        fingerprint1 = base_model.get_fingerprint()
        fingerprint2 = testing_model.get_fingerprint()
        if not isinstance(fingerprint1, torch.Tensor):
            fingerprint1 = torch.tensor(fingerprint1)
        if not isinstance(fingerprint2, torch.Tensor):
            fingerprint2 = torch.tensor(fingerprint2)
        
        # Flatten the tensors if they are multi-dimensional
        fingerprint1 = fingerprint1.flatten().to(self.accelerator.device)
        fingerprint2 = fingerprint2.flatten().to(self.accelerator.device)
        
        # Compute cosine similarity
        cosine_sim = (F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0)) + 1) / 2
        
        # Return the similarity score as a float
        return cosine_sim.item()