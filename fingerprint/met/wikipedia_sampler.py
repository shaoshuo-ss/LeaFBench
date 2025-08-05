import os
import random
import pandas as pd
from datasets import load_dataset
import re


def clean_text(text):
    """
    Clean Wikipedia text by removing unwanted characters and formatting.
    
    Args:
        text (str): Raw text from Wikipedia
        
    Returns:
        str: Cleaned text
    """
    # Remove multiple spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_substring(text, target_length=100):
    """
    Extract a substring of specified length from the given text.
    
    Args:
        text (str): Input text
        target_length (int): Desired length of substring
        
    Returns:
        str: Substring of target_length or None if text is too short
    """
    cleaned_text = clean_text(text)
    
    # Need some buffer for selection
    min_required_length = target_length + 20
    if len(cleaned_text) < min_required_length:
        return None
    
    # Try to find a good starting point that doesn't start mid-word
    max_attempts = 10
    for _ in range(max_attempts):
        start_pos = random.randint(0, len(cleaned_text) - target_length)
        
        # Try to start at word boundary if possible
        if start_pos > 0 and cleaned_text[start_pos - 1] != ' ':
            # Find next space
            next_space = cleaned_text.find(' ', start_pos)
            if next_space != -1 and next_space < len(cleaned_text) - target_length:
                start_pos = next_space + 1
        
        substring = cleaned_text[start_pos:start_pos + target_length]
        
        # Check if substring is valid (contains meaningful content)
        min_valid_length = int(target_length * 0.9)  # Allow 10% tolerance
        min_words = max(10, target_length // 15)  # Estimate minimum words based on length
        if len(substring.strip()) >= min_valid_length and len(substring.split()) >= min_words:
            return substring
    
    # Fallback: just take first target_length characters if available
    if len(cleaned_text) >= target_length:
        return cleaned_text[:target_length]
    
    return None


def create_completion_prompt(text_sample):
    """
    Create a completion prompt from the text sample.
    
    Args:
        text_sample (str): 100-character text sample
        
    Returns:
        str: Formatted prompt for completion task
    """
    prompt_template = (
        "Continue the paragraph. Do not output anything except the continuation to the paragraph. "
        "Start the continuation immediately.\n \"{}\""
    )
    return prompt_template.format(text_sample)


def sample_wikipedia_texts(n_samples=25, text_length=100, cache_path=None, regenerate=False):
    """
    Sample texts from Wikipedia dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        text_length (int): Length of each text sample
        cache_path (str): Path to save/load the samples
        regenerate (bool): Whether to regenerate samples even if cache exists
        
    Returns:
        list: List of dictionaries containing 'text_sample' and 'prompt'
    """
    # Check if cache exists and we don't need to regenerate
    if cache_path and os.path.exists(cache_path) and not regenerate:
        print(f"Loading existing samples from {cache_path}")
        df = pd.read_csv(cache_path)
        return df.to_dict('records')
    
    print("Sampling new texts from Wikipedia dataset...")
    
    try:
        # Load Wikipedia dataset (using the "20220301.en" subset which is commonly available)
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        
        samples = []
        processed_count = 0
        max_attempts = n_samples * 50  # Allow more attempts to find good samples
        
        print(f"Generating {n_samples} samples...")
        
        for i, example in enumerate(dataset):
            if len(samples) >= n_samples:
                break
                
            if i >= max_attempts:
                print(f"Reached maximum attempts ({max_attempts}), stopping with {len(samples)} samples")
                break
            
            text = example.get('text', '')
            if not text:
                continue
            
            # Extract substring of specified length
            text_sample = extract_substring(text, target_length=text_length)
            if text_sample is None:
                continue
            
            # Create completion prompt
            prompt = create_completion_prompt(text_sample)
            
            samples.append({
                'text_sample': text_sample,
                'prompt': prompt
            })
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Generated {len(samples)} valid samples from {processed_count} attempts")
        
        print(f"Successfully generated {len(samples)} samples")
        
        # Save to cache if path provided
        if cache_path and samples:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df = pd.DataFrame(samples)
            df.to_csv(cache_path, index=False)
            print(f"Samples saved to {cache_path}")
        
        return samples
        
    except Exception as e:
        print(f"Error loading Wikipedia dataset: {e}")
        print("Falling back to dummy data generation...")
        return generate_dummy_samples(n_samples, text_length, cache_path)


def generate_dummy_samples(n_samples=25, text_length=100, cache_path=None):
    """
    Generate dummy samples for testing when Wikipedia dataset is not available.
    
    Args:
        n_samples (int): Number of samples to generate
        text_length (int): Length of each text sample
        cache_path (str): Path to save the samples
        
    Returns:
        list: List of dictionaries containing 'text_sample' and 'prompt'
    """
    print("Generating dummy samples for testing...")
    
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet exactly once, making it useful for testing purposes.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for each task.",
        "The sun is a medium-sized star located at the center of our solar system. It provides the energy that drives weather patterns and supports life on Earth.",
        "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and automation.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities have been the main driver of climate change since the 1800s.",
    ]
    
    samples = []
    for i in range(n_samples):
        # Cycle through dummy texts and modify them slightly
        base_text = dummy_texts[i % len(dummy_texts)]
        
        # Add some variation and ensure we have the desired length
        if len(base_text) >= text_length:
            text_sample = base_text[:text_length]
        else:
            # Repeat the text until we have enough length
            repeated_text = base_text
            while len(repeated_text) < text_length:
                repeated_text += " " + base_text
            text_sample = repeated_text[:text_length]
        
        prompt = create_completion_prompt(text_sample)
        
        samples.append({
            'text_sample': text_sample,
            'prompt': prompt
        })
    
    # Save to cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df = pd.DataFrame(samples)
        df.to_csv(cache_path, index=False)
        print(f"Dummy samples saved to {cache_path}")
    
    return samples
