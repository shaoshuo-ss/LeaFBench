from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.reef.generate_activation import load_statements, get_acts
from fingerprint.reef.compute_cka import CKA


class REEFFingerprint(LLMFingerprintInterface):
    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)

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
        self.batch_size = self.config.get('batch_size', 1)
    
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
            self.accelerator.device,
            batch_size=self.batch_size
        )
        return fingerprint
        
    
    def compare_fingerprints(self, base_model, testing_model):
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
        print(f"Base fingerprint shape: {base_fingerprint.shape}")
        testing_fingerprint = testing_model.get_fingerprint()
        base_fingerprint = base_fingerprint.to(self.accelerator.device)
        testing_fingerprint = testing_fingerprint.to(self.accelerator.device)
        cka_value = cka.linear_CKA(base_fingerprint, testing_fingerprint)
        return cka_value.item()