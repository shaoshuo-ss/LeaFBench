import tokenize
from fingerprint.fingerprint_interface import LLMFingerprintInterface
import os
import pandas as pd
from generate_csv import generate_csv, get_prompts_and_targets



class TRAPFingerprint(LLMFingerprintInterface):
    """
    TRAP Fingerprint Class
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        self.n_goals = self.config.get('n_goals', 100)
        self.string_type = self.config.get('string_type', 'number')
        self.string_length = self.config.get('string_length', 3)
        self.prompt_path = self.config.get('prompt_path', None)
        self.filter_tokens_path = self.config.get('filter_tokens_path', None)


    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        if os.path.exists(self.prompt_path):
            df = pd.read_csv(self.prompt_path, dtype={'prompt': str, 'goal': str})
        else:
            df = generate_csv(self.n_goals, self.string_type, self.string_length, self.prompt_path)
        self.prompts = df['prompt'].tolist()
        self.targets = df['target'].tolist()


    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        # only extract fingerprint if the model is pretrained or instruct model
        if model.model_name == model.pretrained_model or model.model_name == model.instruct_model:
            # Step 1: Filter the tokens
            torch_model, tokenizer = model.load_model()
            # Step 2: Using GCG to generate the prefix
        else:
            return 0
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
