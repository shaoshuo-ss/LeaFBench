from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.pdf.parameter_distribution import get_transformer_parameters, get_correlation_coefficient
import torch


class PDFFingerprint(LLMFingerprintInterface):
    """
    Parameter Distribution Fingerprinting method for LLMs.
    This class implements the fingerprinting logic specific to PDF models.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        # Initialize any specific parameters or configurations for PDF fingerprinting
        

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
        torch_model, tokenizer = model.load_model()
        Sq, Sk, Sv, So = get_transformer_parameters(torch_model, self.accelerator)
        fingerprint = torch.vstack([
            Sq.flatten(),
            Sk.flatten(),
            Sv.flatten(),
            So.flatten()
        ])
        return fingerprint
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.
        Calculates correlation coefficients separately for Sq, Sk, Sv, So vectors
        and returns the average as the final similarity score.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Average similarity score between the four parameter vectors.
        """
        # Get fingerprints for both models
        base_fingerprint = base_model.get_fingerprint()
        testing_fingerprint = testing_model.get_fingerprint()

        # Extract the four parameter vectors from the stacked fingerprints
        # The fingerprint is structured as: [Sq, Sk, Sv, So] vertically stacked
        
        base_Sq = base_fingerprint[0]
        base_Sk = base_fingerprint[1]
        base_Sv = base_fingerprint[2]
        base_So = base_fingerprint[3]
        
        testing_Sq = testing_fingerprint[0]
        testing_Sk = testing_fingerprint[1]
        testing_Sv = testing_fingerprint[2]
        testing_So = testing_fingerprint[3]

        # Calculate correlation coefficients for each parameter type
        corr_q = get_correlation_coefficient(base_Sq, testing_Sq)
        corr_k = get_correlation_coefficient(base_Sk, testing_Sk)
        corr_v = get_correlation_coefficient(base_Sv, testing_Sv)
        corr_o = get_correlation_coefficient(base_So, testing_So)
        
        # Return the average of all four correlations
        similarity_score = (corr_q + corr_k + corr_v + corr_o) / 4.0
        return similarity_score