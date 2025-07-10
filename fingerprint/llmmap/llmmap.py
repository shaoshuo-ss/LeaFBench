from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.llmmap.inference import InferenceModel

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