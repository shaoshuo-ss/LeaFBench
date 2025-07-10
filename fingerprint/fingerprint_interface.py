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

    def evaluate(self, train_models, test_models):
        """
        Evaluate the fingerprinting method on a given training and testing model.

        Args:
            train_models: The models used for training.
            test_models: The models used for testing.

        Returns:
            tuple(float, float): The evaluation scores of training and testing models.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def compare(self, fingerprint1, fingerprint2):
        """
        Compare two fingerprints.

        Args:
            fingerprint1 (torch.Tensor): The first fingerprint tensor.
            fingerprint2 (torch.Tensor): The second fingerprint tensor.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
