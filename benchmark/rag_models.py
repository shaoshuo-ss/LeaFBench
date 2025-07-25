import torch
import faiss
import logging
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import time
from benchmark.model_interface import ModelInterface


def load_rag_dataset(dataset_name):
    """
    Load a dataset for RAG from Hugging Face datasets.
    """
    if dataset_name == "squad_v2":
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")
        unique_contexts = set(item['context'] for item in dataset)
        corpus = list(unique_contexts)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return corpus



class RAGModel(ModelInterface):
    """
    A simple implementation of a Retrieval-Augmented Generation (RAG) model.
    This model retrieves relevant documents from a knowledge base and generates answers based on them.
    """
    
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
        logger = logging.getLogger(__name__)
        # load rag corpus
        self.dataset_name = config.get("dataset_name", None)
        if not self.dataset_name:
            raise ValueError("Dataset name must be specified in the configuration.")
        self.corpus = load_rag_dataset(self.dataset_name)

        # load retriver model
        self.retriever_model_name = config.get("retriever_model_name", None)
        if not self.retriever_model_name:
            raise ValueError("Retriever model name must be specified in the configuration.")
        self.retriever_model = SentenceTransformer(self.retriever_model_name, device=accelerator.device)
        self.num_of_retrieved_docs = config.get("num_of_retrieved_docs", 1)

        # Building the FAISS index for the retriever...
        self.corpus_embeddings = self.retriever_model.encode(self.corpus, convert_to_tensor=True, show_progress_bar=True)
        embedding_dim = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)

        self.index.add(self.corpus_embeddings.cpu().numpy())
        self.retriever_model.to("cpu")

    def generate(self, prompts, **kwargs):
        """
        Generate answers for the given prompts using the RAG framework.
        """
        model, tokenizer = self.load_model()
        generation_params = {
            'max_new_tokens': self.params.get('max_new_tokens', 512),
            'temperature': self.params.get('temperature', 0.7),
            'do_sample': self.params.get('do_sample', True),
            'top_p': self.params.get('top_p', 0.9),
            'top_k': self.params.get('top_k', 50),
            'pad_token_id': tokenizer.pad_token_id,
        }
        answers = []
        self.retriever_model.to(self.accelerator.device)
        for prompt in prompts:
            # Retrieve relevant documents
            question_embedding = self.retriever_model.encode([prompt], convert_to_tensor=True)
            distances, indices = self.index.search(question_embedding.cpu().numpy(), self.num_of_retrieved_docs)
            retrieved_contexts = [self.corpus[i] for i in indices[0]]

            # Prepare the prompt with retrieved contexts
            combined_context = "\n---\n".join(retrieved_contexts)
            full_prompt = f"Context:\n{combined_context}\n\nQuestion: {prompt}\n\nAnswer:"

            # Tokenize input prompt
            inputs = tokenizer(
                full_prompt, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=self.params.get('max_input_length', 512),
                padding_side='left'
            )
            # Move inputs to the same device as model
            if self.accelerator is not None:
                # When using accelerator, it handles device placement
                device = self.accelerator.device
            else:
                device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_params)
            
            # Decode the generated tokens
            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answers.append(answer)
        self.retriever_model.to("cpu")
        return answers
