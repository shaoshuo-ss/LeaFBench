import os
from transformers import AutoTokenizer, AutoModel
import torch
import math

# CACHE_DIR = os.environ.get('HF_MODEL_CACHE', None)
    
class Embedding:
    def __init__(self, config, device_map="auto", accelerator=None):
        self.model_name = config['embedding_model_path']
        self.accelerator = accelerator
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Use accelerator for model preparation if available
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = self.accelerator.device
        else:
            # Use device_map if no accelerator
            if device_map == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device_map)
            self.model = self.model.to(self.device)
            
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 8)
        
    def get_embs(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, s, batch_size=None):
        """
        Get embeddings for input text(s) with batch processing.
        
        Args:
            s (str or list): Input text(s) to get embeddings for
            batch_size (int, optional): Batch size for processing. If None, uses self.batch_size
            
        Returns:
            torch.Tensor: Embeddings tensor with shape (num_texts, embedding_dim)
        """
        # Convert single string to list
        if isinstance(s, str):
            s = [s]
        
        # Use provided batch_size or default
        if batch_size is None:
            batch_size = self.batch_size
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(s), batch_size):
            batch_texts = s[i:i+batch_size]
            
            # Tokenize batch
            prompts_tok = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                add_special_tokens=True,
                truncation=True, 
                max_length=self.max_length
            )
            
            # Move to device
            prompts_tok = {k: v.to(self.device) for k, v in prompts_tok.items()}
            
            # Get model output
            with torch.no_grad():  # Add no_grad for inference efficiency
                emb = self.model(**prompts_tok)
                
            # Extract embeddings using existing method
            batch_embeddings = self.get_embs(emb, prompts_tok['attention_mask'])
            all_embeddings.append(batch_embeddings.detach())
        
        # Concatenate all batch embeddings
        return torch.cat(all_embeddings, dim=0)
        
    
# EMBEDDING_MODELS = [
#     ('intfloat/multilingual-e5-large-instruct', Embedding),
# ]

# def load_model(embedding_model_path):
#     model = Embedding(embedding_model_path)
#     return model