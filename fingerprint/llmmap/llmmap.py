from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.llmmap.inference import InferenceModel
import torch
import torch.nn.functional as F

class LLMmap(LLMFingerprintInterface):
    """
    LLMmap fingerprinting method.
    This class implements the LLMmap fingerprinting technique.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config, accelerator)
        

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods.
        """
        self.inference_model = InferenceModel(config=self.config, accelerator=self.accelerator)
        if self.inference_model.model is None:
            # training inference model
            self.inference_model.train_inference_model(train_models)

    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model.

        Args:
            model: The model to extract the fingerprint.
        
        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        if not hasattr(self, 'inference_model'):
            raise RuntimeError("Inference model is not prepared. Call prepare() first.")
        
        # Generate the fingerprint using the inference model
        return self.inference_model.get_fingerprint(model)
    
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