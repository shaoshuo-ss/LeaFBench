import torch



class LLMFingerprintInterface:
    """
    Interface for LLM fingerprinting.
    """

    def __init__(self, config=None, accelerator=None):
        self.config = config
        self.accelerator = accelerator

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        pass
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def compare(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
