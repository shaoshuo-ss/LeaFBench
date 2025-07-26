from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.gradient.calculate_gradient import load_statements, get_gradients_stats
import torch
import torch.nn.functional as F



class GradientFingerprint(LLMFingerprintInterface):
    """
    Gradient Fingerprinting method for LLMs.
    This class implements the fingerprinting logic specific to gradient-based models.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        # Initialize any specific parameters or configurations for gradient fingerprinting

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        dataset_path = self.config.get('dataset_path', None)
        num_samples = self.config.get('num_samples', 200)
        self.statements = load_statements(dataset_path)[:num_samples]
        self.batch_size = self.config.get('batch_size', 1)
        self.fingerprint_dim = self.config.get('fingerprint_dim', 16)  # Default to 16 dimensions for gradient stats
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model.

        Args:
            model: The input model to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        # Implementation for generating gradient fingerprints
        torch_model, tokenizer = model.load_model()
        fingerprint = get_gradients_stats(
            torch_model, tokenizer, self.statements, 
            batch_size=self.batch_size
        )
        
        return fingerprint
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model: The base model to compare against.
            testing_model: The model to compare.

        Returns:
            float: Similarity score between the two models.
        """
        # Implementation for comparing gradient fingerprints
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
        print(fingerprint1)
        
        # Compute cosine similarity
        cosine_sim = (F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0)) + 1) / 2
        
        # Return the similarity score as a float
        return cosine_sim.item()