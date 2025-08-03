import torch
from benchmark.model_interface import ModelInterface
from benchmark.watermark.extended_watermark_processor import WatermarkLogitsProcessor
from transformers import LogitsProcessorList


class WatermarkModel(ModelInterface):
    """
    Base model class that inherits from ModelInterface.
    This class can be used to implement common functionality for all models.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
        self.gamma = config.get('gamma', 0.25)
        self.delta = config.get('delta', 2.0)
        self.seeding_scheme = config.get('seeding_scheme', 'selfhash')
    
    def generate(self, prompts, **kwargs):
        """
        Generate text for given prompts.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of generated text strings
        """
        model, tokenizer = self.load_model()

        watermark_processor = WatermarkLogitsProcessor(
            vocab=tokenizer.get_vocab().values(),
            gamma=self.gamma,
            delta=self.delta,
            seeding_scheme=self.seeding_scheme
        )
        
        # Default generation parameters
        generation_params = {
            'max_new_tokens': self.params.get('max_new_tokens', 512),
            'temperature': self.params.get('temperature', 0.7),
            'do_sample': self.params.get('do_sample', True),
            'top_p': self.params.get('top_p', 0.9),
            'top_k': self.params.get('top_k', 50),
            'pad_token_id': tokenizer.pad_token_id,
            'logits_processor': LogitsProcessorList([watermark_processor])
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
    
    def generate_logits(self, prompts, **kwargs):
        """
        Generate logits for given prompts.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional parameters
        
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        # if self.model is None:
        model, tokenizer = self.load_model()
        
        # Tokenize input prompts
        inputs = tokenizer(
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
            device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get logits from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Apply temperature if specified
        temperature = kwargs.get('temperature', 1.0)
        if temperature != 1.0:
            logits = logits / temperature
        
        return logits
