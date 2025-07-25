import torch
from benchmark.base_models import BaseModel

class AdapterModel(BaseModel):
    """
    Base model class that inherits from ModelInterface.
    This class can be used to implement common functionality for all models.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
    
    def load_model(self):
        """
        Load adapter model. Need to pass the type as "adapter" to load the adapter model.
        """
        # Use model pool to get the model (singleton pattern)
        model = self.model_pool.get_model(self.base_model, type="adapter")
        # Get tokenizer from model pool as well
        tokenizer = self.model_pool.get_tokenizer(self.base_model)
        return model, tokenizer
