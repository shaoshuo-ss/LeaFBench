import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from accelerate import cpu_offload

from .utility import read_pickle
from .embedding_model import Embedding
from .inference_model_archs import InferenceModelLLMmap
    

class InferenceModel:
            
    def __init__(self, config, accelerator=None):
        
        self.config = config
        self.accelerator = accelerator
        self.logger = logging.getLogger(__name__)
        self.logger.info("Building Inference Model...")

        self.queries = self.config['queries']
        self.logger.info(f"Queries: {self.queries}")

        self.logger.info("\tLoading Embedding Model...")
        self.emb_model = Embedding(self.config['embedding_model'], accelerator=accelerator)
        self.logger.info("\tPre-computing Queries embeddings...")
        self.emb_queries = self.emb_model.get_embeddings(self.queries)
        self.logger.info("Model ready for inference.")

        self.logger.info("\tLoading Inference Model...")
        self.model_path = self.config['inference_model']['inference_model_path']
        if os.path.exists(self.model_path) and not self.config.get("retrain_inference_model", False):
            self.logger.info(f"\tLoading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.model = InferenceModelLLMmap(config['inference_model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Use accelerator to prepare model if available
            if self.accelerator is not None:
                self.model = self.accelerator.prepare(self.model)
            
            self.model.eval()  # Set to evaluation mode
        else:
            self.logger.info(f"\tModel path {self.model_path} does not exist. Need to train the model.")
            self.model = None

    def __call__(self, answers, train=False):
        traces_tensor = self.get_embeddings(answers)

        if train:
            output = self.model(traces_tensor)
        else:
            with torch.no_grad():
                output = self.model(traces_tensor)
        
        return output[0]
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model.

        Args:
            model: The model to extract the fingerprint.
        
        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Inference model is not prepared. Call prepare() first.")
        
        # Generate the fingerprint using the inference model
        with torch.no_grad():
            answers = model.generate(self.queries)
        # Get embeddings for these answers
            embeddings = self.get_embeddings(answers)  # Shape: (1, num_queries, emb_size*2)
            fingerprint = self.model(embeddings)
        # model.fingerprint = fingerprint.squeeze(0)  # Remove batch dimension

        return fingerprint.squeeze(0)
    
    def get_embeddings(self, answers):
        if len(answers) != len(self.queries):
            raise Exception(f"Model supports {self.queries} queries, {len(answers)} answers provided")
        answers = [self._preprocess_answers(answer) for answer in answers]
        emb_outs = self.emb_model.get_embeddings(answers) 
        traces = torch.concatenate((self.emb_queries, emb_outs), 1).unsqueeze(0)  # Add batch dimension
        
        # Convert to PyTorch tensor
        traces_tensor = traces.float().to(self.emb_model.device)
        return traces_tensor
        
    def _preprocess_answers(self, out):
        return out[:self.config.get('max_answer_chars', 650)]
    
    def train_inference_model(self, train_models):
        """
        Train the inference model using contrastive learning.
        
        Args:
            train_models (list): List of ModelInterface instances for training
        """
        self.train_config = self.config['inference_model']['train_config']
        self.model = InferenceModelLLMmap(self.config['inference_model'])
        
        # Training parameters
        num_repeats_per_model = self.train_config.get('num_repeats_per_model', 5)
        epochs = self.train_config.get('epochs', 100)
        learning_rate = self.train_config.get('learning_rate', 1e-4)
        batch_size = self.train_config.get('batch_size', 16)
        temperature = self.train_config.get('temperature', 0.07)

        self.logger.info(f"Training with {len(train_models)} models, {num_repeats_per_model} repeats per model")

        # Step 1: Generate answers from all models
        self.logger.info("Step 1: Generating answers from all models...")
        all_embeddings = []
        all_labels = []
        
        for model_idx, (model_name, model) in enumerate(tqdm(train_models.items(), desc="Processing models")):
            model_embeddings = []
            self.logger.info(f"Processing model: {model_name}")
            
            for sample_idx in tqdm(range(num_repeats_per_model), desc=f"Generating answers for the model {model_name}"):
                # Generate answers for all queries
                answers = model.generate(self.queries)
                
                # Get embeddings for these answers
                embeddings = self.get_embeddings(answers)  # Shape: (1, num_queries, emb_size*2)
                model_embeddings.append(embeddings.squeeze(0))  # Remove batch dimension
            
            # Stack embeddings for this model: (num_samples, num_queries, emb_size*2)
            model_embeddings = torch.stack(model_embeddings)
            all_embeddings.append(model_embeddings)
            all_labels.extend([model_idx] * num_repeats_per_model)
            
            # Release GPU memory after processing this model
            # model.unload_model(to_cpu=True)
            # self.logger.info(f"Released GPU memory for model: {model_name}")
        
        # Concatenate all embeddings: (total_samples, num_queries, emb_size*2)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.tensor(all_labels)
        
        self.logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Step 2: Train the model with contrastive loss
        self.logger.info("Step 2: Training inference model...")
        
        # Move model and data to device
        if self.accelerator is not None:
            device = self.accelerator.device
            self.model = self.accelerator.prepare(self.model)
        else:
            device = self.emb_model.device
            self.model = self.model.to(device)
            
        all_embeddings = all_embeddings.to(device)
        all_labels = all_labels.to(device)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Prepare optimizer with accelerator if available
        if self.accelerator is not None:
            optimizer = self.accelerator.prepare(optimizer)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(all_embeddings, all_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare dataloader with accelerator if available
        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_embeddings, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                features = self.model(batch_embeddings)  # (batch_size, feature_size)
                
                # Compute contrastive loss
                loss = self._contrastive_loss(features, batch_labels, temperature)
                
                # Backward pass
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            # if (epoch + 1) % 10 == 0:
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save the trained model
        model_save_path = self.config['inference_model']['inference_model_path']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Extract the actual model from accelerator wrapper if needed
        if self.accelerator is not None and hasattr(self.model, 'module'):
            # If wrapped by DistributedDataParallel, save the underlying model
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        # Save model state dict in a format compatible with loading
        checkpoint = {'model_state_dict': model_to_save.state_dict()}
        torch.save(checkpoint, model_save_path)
        self.logger.info(f"Model saved to {model_save_path}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _contrastive_loss(self, features, labels, temperature=0.07):
        """
        Compute contrastive loss (InfoNCE loss).
        
        Args:
            features (torch.Tensor): Feature embeddings of shape (batch_size, feature_size)
            labels (torch.Tensor): Labels of shape (batch_size,)
            temperature (float): Temperature parameter for scaling
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask_no_diag = mask - torch.eye(batch_size, device=mask.device)
        
        # Apply log-softmax
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        
        # Compute positive loss
        positive_loss = -(log_prob * mask_no_diag).sum(dim=1)
        
        # Normalize by number of positive pairs
        num_positive_pairs = mask_no_diag.sum(dim=1)
        num_positive_pairs = torch.clamp(num_positive_pairs, min=1.0)  # Avoid division by zero
        positive_loss = positive_loss / num_positive_pairs
        
        return positive_loss.mean()
                
# class InferenceModel_closed(InferenceModel):

#     def __call__(self, answers):
#         logits = super().__call__(answers)
#         # Convert to PyTorch tensor, apply softmax, and convert back to numpy
#         logits_tensor = torch.from_numpy(logits).float()
#         p = F.softmax(logits_tensor, dim=-1).numpy()
#         return p
    
#     def print_result(self, probabilities, k=5):
#         if k < 1:
#             raise ValueError("k must be at least 1")
#         if k > len(probabilities):
#             raise ValueError("k cannot be greater than the number of classes")

#         sorted_indices = np.argsort(probabilities)[::-1]
#         top_k_indices = sorted_indices[:k]
#         top_k_probs = probabilities[top_k_indices]

#         print("Prediction:\n")
#         for i, (index, prob) in enumerate(zip(top_k_indices, top_k_probs)):
#             if prob < 0.001:
#                 prob_str = f"{prob:.1e}"
#             else:
#                 prob_str = f"{prob:.4f}"

#             if i == 0:  # Top-1 class
#                 print(f"\t[Pr: {prob_str}] \t--> {self.label_map[index]} <--")
#             else:
#                 print(f"\t[Pr: {prob_str}] \t{self.label_map[index]}")
                
# class InferenceModel_open(InferenceModel):

#     def __init__(self, *args, **kargs):
#         super().__init__(*args, **kargs)
#         self.DB_labels, self.DB = self.config['DB_templates']
#         self.distance_fn = self.config['distance_fn']
        
#     def __call__(self, answers):
#         emb = super().__call__(answers)
#         distances = cdist([emb], self.DB, metric=self.distance_fn)[0]
#         return distances

#     def print_result(self, distances, k=5):
#         if k < 1:
#             raise ValueError("k must be at least 1")
#         if k > len(distances):
#             raise ValueError("k cannot be greater than the number of classes")

#         sorted_indices = np.argsort(distances)
#         top_k_indices = sorted_indices[:k]
#         top_k_probs = distances[top_k_indices]

#         print("Prediction:\n")
#         for i, (index, dist) in enumerate(zip(top_k_indices, top_k_probs)):
#             if dist < 0.001:
#                 dist_str = f"{dist:.1e}"
#             else:
#                 dist_str = f"{dist:.4f}"

#             if i == 0:  # Top-1 class
#                 print(f"\t[Distance: {dist_str}] \t--> {self.label_map[index]} <--")
#             else:
#                 print(f"\t[Distance: {dist_str}] \t{self.label_map[index]}")

                

        
