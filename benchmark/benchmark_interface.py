import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

# model interface
class ModelInterface:
    """
    Model interface, any model in the benchmark should inherit from this class.
    This class is mainly used to store model metadata, while the actual model loading 
    and inference logic can be implemented in subclasses.
    """
    def __init__(self, config):
        self.model_family = config.get("model_family", None)
        self.pretrained_model = config.get("pretrained_model", None)
        self.base_model = config.get("base_model", None)
        self.model_name = config.get("model_name", None)
        self.model_path = config.get("model_path", None)
        self.model = None
        self.fingerprint = None
        self.tokenizer = None
        self.params = config.get("params", None)

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

    def __str__(self):
        return f"Model(name={self.model_name}, family={self.model_family}, base_model={self.base_model})"

class BaseModel(ModelInterface):
    """
    Base model class that inherits from ModelInterface.
    This class can be used to implement common functionality for all models.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config)
        self.model_pool = model_pool
        self.accelerator = accelerator

    def is_model_loaded(self):
        """
        Check if the model is currently loaded.
        """
        return self.model is not None
    
    def _load_model(self):
        """
        Load the model weights and prepare it for inference.
        Uses the model pool if available, otherwise loads directly.
        Uses accelerator for proper multi-GPU management if available.
        """
        # Use model pool to get the model (singleton pattern)
        self.model = self.model_pool.get_model(self.base_model)
        # Get tokenizer from model pool as well
        self.tokenizer = self.model_pool.get_tokenizer(self.model_name)
    
    def generate(self, prompts, **kwargs):
        """
        Generate text for given prompts.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of generated text strings
        """
        if self.model is None:
            self._load_model()
        
        # Default generation parameters
        generation_params = {
            'max_new_tokens': kwargs.get('max_new_tokens', 256),
            'temperature': kwargs.get('temperature', 0.7),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'pad_token_id': self.tokenizer.pad_token_id,
        }

        system_prompt = kwargs.get('system_prompt', None)
        if system_prompt is not None:
            # If a system prompt is provided, prepend it to each prompt
            prompts = [system_prompt.replace("{{user_input}}", prompt) for prompt in prompts]
        
        # Tokenize input prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=kwargs.get('max_input_length', 256)
        )
        
        # Move inputs to the same device as model
        if self.accelerator is not None:
            # When using accelerator, it handles device placement
            device = self.accelerator.device
        else:
            device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )
        
        # Decode generated text
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_logits(self, prompts, **kwargs):
        """
        Generate logits for given prompts.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional parameters
        
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        if self.model is None:
            self._load_model()
        
        # Tokenize input prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=kwargs.get('max_input_length', 512)
        )
        
        # Move inputs to the same device as model
        if self.accelerator is not None:
            # When using accelerator, it handles device placement
            device = self.accelerator.device
        else:
            device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get logits from model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply temperature if specified
        temperature = kwargs.get('temperature', 1.0)
        if temperature != 1.0:
            logits = logits / temperature
        
        return logits
