import torch
from benchmark.model_interface import ModelInterface
from transformers import AutoModelForCausalLM, AutoTokenizer

class InputParaphraseModel(ModelInterface):
    """
    Adversarial model that paraphrases input prompts before generation.
    This class uses a small paraphrase model to rewrite prompts while maintaining their meaning,
    then feeds the paraphrased prompts to the main generation model.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
        # Configuration for the paraphrase model
        self.paraphrase_model_name = self.params.get('paraphrase_model', 'Qwen/Qwen3-0.6B')
        self.paraphrase_model = None
        self.paraphrase_tokenizer = None
    
    def load_paraphrase_model(self):
        """
        Load the paraphrase model for input rewriting.
        """
        if self.paraphrase_model is None:
            # Load a small LLM model for paraphrasing
            self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(self.paraphrase_model_name)
            self.paraphrase_model = AutoModelForCausalLM.from_pretrained(self.paraphrase_model_name)

        return self.paraphrase_model, self.paraphrase_tokenizer
    
    def paraphrase_prompts(self, prompts):
        """
        Paraphrase the input prompts while maintaining their meaning.
        
        Args:
            prompts (list): List of input prompt strings
            
        Returns:
            list: List of paraphrased prompt strings
        """
        paraphrase_model, paraphrase_tokenizer = self.load_paraphrase_model()
        
        paraphrase_model = paraphrase_model.to(self.accelerator.device if self.accelerator else 'cpu')
        paraphrased_prompts = []
        for prompt in prompts:
            # Prepare input for T5 model (add task prefix)
            input_text = f"Please paraphrase the following text while maintaining the meaning. Only answer the paraphrased text: '{prompt}'"
            
            # Tokenize
            inputs = paraphrase_tokenizer.encode(
                input_text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True
            )
            
            # Move to device
            if self.accelerator is not None:
                device = self.accelerator.device
            else:
                device = paraphrase_model.device
            inputs = inputs.to(device)
            
            # Generate paraphrase
            # with torch.no_grad():
            outputs = paraphrase_model.generate(
                inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
            
            # Decode the paraphrased text
            paraphrased_text = paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            paraphrased_prompts.append(paraphrased_text)
        
        paraphrase_model = paraphrase_model.cpu()  # Move back to CPU after generation
        return paraphrased_prompts
    
    def generate(self, prompts, **kwargs):
        """
        Generate text for given prompts with input paraphrasing.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of generated text strings
        """
        # Step 1: Paraphrase input prompts
        # if self.params.get('enable_paraphrase', True):
        print(f"Original prompts: {prompts[:2]}...")  # Show first 2 for debugging
        paraphrased_prompts = self.paraphrase_prompts(prompts)
        print(f"Paraphrased prompts: {paraphrased_prompts[:2]}...")  # Show first 2 for debugging
        # else:
        #     paraphrased_prompts = prompts
        
        # Step 2: Load main generation model
        model, tokenizer = self.load_model()
        
        # Default generation parameters
        generation_params = {
            'max_new_tokens': self.params.get('max_new_tokens', 512),
            'temperature': self.params.get('temperature', 0.7),
            'do_sample': self.params.get('do_sample', True),
            'top_p': self.params.get('top_p', 0.9),
            'top_k': self.params.get('top_k', 50),
            'pad_token_id': tokenizer.pad_token_id,
        }

        # Prepare messages for chat template
        system_prompt = self.params.get('system_prompt', None)
        
        # Convert prompts to chat format
        chat_messages_list = []
        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            chat_messages_list.append(messages)
        
        # Apply chat template and tokenize
        tokenized_prompts = []
        for messages in chat_messages_list:
            # Apply chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                if system_prompt is not None:
                    formatted_prompt = f"{system_prompt}\n\nUser: {messages[-1]['content']}\n\nAssistant:"
                else:
                    formatted_prompt = f"User: {messages[-1]['content']}\n\nAssistant:"
            tokenized_prompts.append(formatted_prompt)
        
        # Tokenize input prompts
        inputs = tokenizer(
            tokenized_prompts, 
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
        # Special handling for Gemma-2 models to avoid cache device mismatch
        model_name_lower = model.__class__.__name__.lower()
        config_name_lower = getattr(model.config, 'model_family', '').lower()
        if "gemma" in model_name_lower or "gemma" in config_name_lower:
            # For Gemma models, disable cache to avoid device mismatch issues
            generation_params['use_cache'] = False
        
        # with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params
        )
        
        # Decode generated text
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts


class OutputPerturbationModel(ModelInterface):
    """
    Adversarial model that perturbs the output logits of a given model.
    This class adds Gaussian noise to the logits during generation to test model robustness.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
    
    def generate_with_logits_perturbation(self, model, tokenizer, inputs, generation_params, device):
        """
        Generate text with Gaussian noise added to logits.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            inputs: Tokenized input
            generation_params: Generation parameters
            device: Device to run on
            
        Returns:
            Generated token sequences
        """
        batch_size = inputs['input_ids'].shape[0]
        max_new_tokens = generation_params.get('max_new_tokens', 512)
        temperature = generation_params.get('temperature', 0.7)
        do_sample = generation_params.get('do_sample', True)
        top_p = generation_params.get('top_p', 0.9)
        top_k = generation_params.get('top_k', 50)
        
        # Noise parameters
        noise_variance = self.params.get('noise_variance', 0.1)  # delta parameter
        
        # Initialize generation
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        # Store generated sequences
        generated_sequences = input_ids.clone()
        
        # Generate tokens one by one
        for step in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = model(
                    input_ids=generated_sequences,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Get logits for the last token
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Add Gaussian noise to logits
            noise = torch.normal(
                mean=0.0, 
                std=noise_variance, 
                size=logits.shape,
                device=device
            )
            perturbed_logits = logits + noise
            
            # Apply temperature scaling
            if temperature != 1.0:
                perturbed_logits = perturbed_logits / temperature
            
            # Apply sampling strategy
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(perturbed_logits, top_k, dim=-1)
                    perturbed_logits = torch.full_like(perturbed_logits, float('-inf'))
                    perturbed_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(perturbed_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    perturbed_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(perturbed_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(perturbed_logits, dim=-1)
            
            # Append new tokens
            generated_sequences = torch.cat([generated_sequences, next_tokens.unsqueeze(1)], dim=1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Check for EOS token
            if tokenizer.eos_token_id is not None:
                finished = (next_tokens == tokenizer.eos_token_id)
                if finished.all():
                    break
        
        return generated_sequences
    
    def generate(self, prompts, **kwargs):
        """
        Generate text for given prompts with logits perturbation.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of generated text strings
        """
        model, tokenizer = self.load_model()
        
        # Default generation parameters
        generation_params = {
            'max_new_tokens': self.params.get('max_new_tokens', 512),
            'temperature': self.params.get('temperature', 0.7),
            'do_sample': self.params.get('do_sample', True),
            'top_p': self.params.get('top_p', 0.9),
            'top_k': self.params.get('top_k', 50),
            'pad_token_id': tokenizer.pad_token_id,
        }

        # Prepare messages for chat template
        system_prompt = self.params.get('system_prompt', None)
        
        # Convert prompts to chat format
        chat_messages_list = []
        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            chat_messages_list.append(messages)
        
        # Apply chat template and tokenize
        tokenized_prompts = []
        for messages in chat_messages_list:
            # Apply chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                if system_prompt is not None:
                    formatted_prompt = f"{system_prompt}\n\nUser: {messages[-1]['content']}\n\nAssistant:"
                else:
                    formatted_prompt = f"User: {messages[-1]['content']}\n\nAssistant:"
            tokenized_prompts.append(formatted_prompt)
        
        # Tokenize input prompts
        inputs = tokenizer(
            tokenized_prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.params.get('max_input_length', 512),
            padding_side='left'
        )

        # Generate text
        # Special handling for Gemma-2 models to avoid cache device mismatch
        model_name_lower = model.__class__.__name__.lower()
        config_name_lower = getattr(model.config, 'model_family', '').lower()
        if "gemma" in model_name_lower or "gemma" in config_name_lower:
            # For Gemma models, disable cache to avoid device mismatch issues
            generation_params['use_cache'] = False

        # Move inputs to the same device as model
        if self.accelerator is not None:
            # When using accelerator, it handles device placement
            device = self.accelerator.device
        else:
            device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate text with logits perturbation
        outputs = self.generate_with_logits_perturbation(
            model, tokenizer, inputs, generation_params, device
        )
        
        # Decode generated text
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
