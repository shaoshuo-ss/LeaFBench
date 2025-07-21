# model interface
class ModelInterface:
    """
    Model interface, any model in the benchmark should inherit from this class.
    This class is mainly used to store model metadata, while the actual model loading 
    and inference logic can be implemented in subclasses.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        self.model_family = config.get("model_family", None)
        self.pretrained_model = config.get("pretrained_model", None)
        self.instruct_model = config.get("instruct_model", None)
        self.base_model = config.get("base_model", None)
        self.model_name = config.get("model_name", None)
        self.model_path = config.get("model_path", None)
        self.type = config.get("type", None)
        self.model = None
        self.fingerprint = None
        self.tokenizer = None
        self.params = config.get("params", None)
        self.model_pool = model_pool
        self.accelerator = accelerator

    def _load_model(self):
        """
        Load the model weights and prepare it for inference.
        Uses the model pool if available, otherwise loads directly.
        Uses accelerator for proper multi-GPU management if available.
        """
        # Use model pool to get the model (singleton pattern)
        model = self.model_pool.get_model(self.base_model)
        # Get tokenizer from model pool as well
        tokenizer = self.model_pool.get_tokenizer(self.base_model)
        return model, tokenizer
    
    def generate(self, prompts, **kwargs):
        """
        Model generation method (simulated).
        In actual applications, this would contain complex logic for loading model weights 
        and executing inference.
        """
        pass

    def generate_logits(self, prompts, **kwargs):
        """
        Model generation method that returns logits (simulated).
        This is a placeholder for actual model inference logic.
        """
        # In actual applications, this would return logits from the model.
        pass

    def set_fingerprint(self, fingerprint):
        """
        Set the fingerprint for the model.
        """
        self.fingerprint = fingerprint

    def get_fingerprint(self):
        """
        Get the fingerprint of the model.
        """
        return self.fingerprint

    def __str__(self):
        return f"Model(name={self.model_name}, family={self.model_family}, base_model={self.base_model})"
