import torch
import torch.nn.functional as F
from typing import List

class SEFFingerprintHelper:
    def __init__(self, config, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.fingerprint_aggregation = config.get('fingerprint_aggregation', 'mean')
        self.generation_batch_size = config.get('generation_batch_size', 8)
        self.embedding_batch_size = config.get('embedding_batch_size', 16)
        self.similarity_metric = config.get('similarity_metric', 'cosine')
    
    def generate_responses_batch(self, model, questions: List[str]) -> List[str]:
        """
        Generate responses for a list of questions using batch processing.
        
        Args:
            model: The model to generate responses
            questions: List of questions
            
        Returns:
            List of generated responses
        """
        responses = []
        batch_size = self.generation_batch_size
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_responses = model.generate(batch_questions)
            responses.extend(batch_responses)
        
        return responses
    
    def get_sentence_embeddings_batch(self, embedding_model, texts: List[str]) -> torch.Tensor:
        """
        Get sentence embeddings for a list of texts using batch processing.
        
        Args:
            embedding_model: SentenceTransformer model
            texts: List of texts to embed
            
        Returns:
            torch.Tensor: Embeddings tensor of shape [n, embedding_dim]
        """
        # SentenceTransformer already supports batch processing internally
        embeddings = embedding_model.encode(
            texts, 
            convert_to_tensor=True,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False
        )
        
        # Move embeddings to accelerator device if available
        if self.accelerator is not None:
            embeddings = embeddings.to(self.accelerator.device)
            
        return embeddings
    
    def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate embeddings based on the configured aggregation method.
        
        Args:
            embeddings: Input embeddings tensor of shape [n, embedding_dim]
            
        Returns:
            torch.Tensor: Aggregated embeddings
        """
        # Ensure embeddings are on the correct device
        if self.accelerator is not None:
            embeddings = embeddings.to(self.accelerator.device)
            
        if self.fingerprint_aggregation == 'mean':
            return torch.mean(embeddings, dim=0, keepdim=True)  # [1, embedding_dim]
        elif self.fingerprint_aggregation == 'original':
            return embeddings  # [n, embedding_dim]
        elif self.fingerprint_aggregation == 'max':
            return torch.max(embeddings, dim=0, keepdim=True)[0]  # [1, embedding_dim]
        elif self.fingerprint_aggregation == 'sum':
            return torch.sum(embeddings, dim=0, keepdim=True)  # [1, embedding_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {self.fingerprint_aggregation}")
    
    def compute_cosine_similarity(self, fingerprint1: torch.Tensor, fingerprint2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two fingerprints and scale to [0,1] range.
        
        Args:
            fingerprint1: First fingerprint tensor
            fingerprint2: Second fingerprint tensor
            
        Returns:
            float: Cosine similarity score in [0,1] range
        """
        # Ensure both fingerprints are tensors
        if not isinstance(fingerprint1, torch.Tensor):
            fingerprint1 = torch.tensor(fingerprint1)
        if not isinstance(fingerprint2, torch.Tensor):
            fingerprint2 = torch.tensor(fingerprint2)
        
        # Flatten the tensors if they are multi-dimensional
        fingerprint1 = fingerprint1.flatten()
        fingerprint2 = fingerprint2.flatten()
        
        # Compute cosine similarity and scale from [-1,1] to [0,1]
        cosine_sim = F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0))
        similarity_score = (cosine_sim + 1) / 2
        
        return similarity_score.item()
    
    def compute_similarity(self, fingerprint1: torch.Tensor, fingerprint2: torch.Tensor) -> float:
        """
        Compute similarity between two fingerprints using the configured metric.
        
        Args:
            fingerprint1: First fingerprint tensor
            fingerprint2: Second fingerprint tensor
            
        Returns:
            float: Similarity score in [0,1] range
        """
        # Ensure both fingerprints are tensors on the correct device
        device = self.accelerator.device if self.accelerator is not None else 'cpu'
        
        if not isinstance(fingerprint1, torch.Tensor):
            fingerprint1 = torch.tensor(fingerprint1, dtype=torch.float32, device=device)
        else:
            fingerprint1 = fingerprint1.to(device)
            
        if not isinstance(fingerprint2, torch.Tensor):
            fingerprint2 = torch.tensor(fingerprint2, dtype=torch.float32, device=device)
        else:
            fingerprint2 = fingerprint2.to(device)
        
        # Flatten the tensors if they are multi-dimensional
        fingerprint1 = fingerprint1.flatten()
        fingerprint2 = fingerprint2.flatten()
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity: [-1,1] -> [0,1]
            cosine_sim = F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0))
            return ((cosine_sim + 1) / 2).item()
            
        elif self.similarity_metric == 'correlation':
            # Pearson correlation coefficient: [-1,1] -> [0,1]
            mean1, mean2 = fingerprint1.mean(), fingerprint2.mean()
            centered1 = fingerprint1 - mean1
            centered2 = fingerprint2 - mean2
            
            numerator = (centered1 * centered2).sum()
            denominator = (centered1.pow(2).sum() * centered2.pow(2).sum()).sqrt()
            
            if denominator == 0:
                correlation = torch.tensor(0.0, device=device)
            else:
                correlation = numerator / denominator
                
            return ((correlation + 1) / 2).item()
            
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance -> similarity: [0,inf] -> [0,1]
            distance = torch.norm(fingerprint1 - fingerprint2, p=2)
            max_distance = torch.norm(fingerprint1) + torch.norm(fingerprint2)
            if max_distance == 0:
                return 1.0
            similarity = 1 - (distance / max_distance)
            return max(0.0, similarity.item())
            
        elif self.similarity_metric == 'manhattan':
            # Manhattan distance -> similarity: [0,inf] -> [0,1]
            distance = torch.norm(fingerprint1 - fingerprint2, p=1)
            max_distance = torch.norm(fingerprint1, p=1) + torch.norm(fingerprint2, p=1)
            if max_distance == 0:
                return 1.0
            similarity = 1 - (distance / max_distance)
            return max(0.0, similarity.item())
            
        elif self.similarity_metric == 'dot_product':
            # Normalized dot product: [0,1]
            norm1 = torch.norm(fingerprint1)
            norm2 = torch.norm(fingerprint2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            normalized_dot = torch.dot(fingerprint1, fingerprint2) / (norm1 * norm2)
            return max(0.0, normalized_dot.item())
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
