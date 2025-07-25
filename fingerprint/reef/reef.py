from fingerprint.fingerprint_interface import FingerprintInterface
from .generate_activation import load_statements, get_acts
from .compute_cka import CKA


class REEF(FingerprintInterface):
    def __init__(self):
        super().__init__()

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        dataset_path = self.config.get('dataset_path', None)
        num_samples = self.config.get('num_samples', 200)
        self.layers = self.config.get('layers', 18)
        self.statements = load_statements(dataset_path)[:num_samples]
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        torch_model, tokenizer = model.load_model()
        fingerprint = get_acts(
            self.statements, tokenizer, torch_model, 
            model.model_family, 
            self.layers, 
            self.accelerator.device
        )
        return fingerprint
        
    
    def compare(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        cka = CKA(self.accelerator.device)
        base_fingerprint = base_model.get_fingerprint()
        testing_fingerprint = testing_model.get_fingerprint()
        cka_value = cka.linear_CKA(base_fingerprint, testing_fingerprint)
        return cka_value.item()