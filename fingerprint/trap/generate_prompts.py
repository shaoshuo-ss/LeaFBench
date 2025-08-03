import random
import string
import pandas as pd
import os
import nanogcg


def generate_random_string(string_length, string_type, lower_case_only=True):
    if string_type == 'string':
        # Define the set of ASCII letters (both lowercase and uppercase)
        ascii_letters = string.ascii_letters
        if lower_case_only:
            ascii_letters = ascii_letters[:26]
    elif 'number' in string_type:
        ascii_letters = string.digits
    else:
        raise NotImplementedError(f'String type not supported: {string_type}')
    # Generate a random string of the specified size
    random_string = ''.join(random.choice(ascii_letters) for _ in range(string_length))
    return random_string

def get_prompt_target(string_length, string_type, random_string):
    if string_type == 'number':
        if string_length == 1:
            user_prompt = f"Write a random digit. Your reply should only contain the random digit."
            target = f"Sure, here is a random digit: {random_string}"
        else:
            user_prompt = f"Write a random string composed of {string_length} digits. Your reply should only contain the random string."
            target = f"Sure, here is a random string of {string_length} digits: {random_string}"
    elif string_type == 'number_simpletarget':
        if string_length == 1:
            user_prompt = f"Write a random digit. Your reply should only contain the random digit."
            target = f"{random_string}"
        else:
            user_prompt = f"Write a random string composed of {string_length} digits. Your reply should only contain the random string."
            target = f"{random_string}"
    elif string_type == 'string':
        user_prompt = f"Write a random string composed of {string_length} lower case letters. Your reply should only contain the random string."
        target = f'Sure, here is a random string of {string_length} lowercase letters: {random_string}'
    else:
        raise NotImplementedError('String type not supported')
    return {'prompt': user_prompt, 'target': target, 'string_target': random_string}

def generate_csv(n_goals, string_type, string_length, csv_path):
    """
    Generate a CSV file with random strings and their corresponding prompts and targets.

    Args:
        n_goals (int): Number of goals to generate.
        string_type (str): Type of string to generate ('string', 'number', etc.).
        string_length (int): Length of the generated strings.
        method (str): Method for generating strings ('random', etc.).
        csv_path (str): Path to save the generated CSV file.
    """
    target_string_list = [generate_random_string(string_length=string_length, string_type=string_type) for _ in range(n_goals)]
    data = [get_prompt_target(string_length=string_length, string_type=string_type, random_string=target_string_list[i]) for i in range(n_goals)]
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def generate_adversarial_suffix(model, tokenizer, prompts, targets, config):
    """
    Generate adversarial prefixes using GCG.

    Args:
        model: The model to use for generating prefixes.
        tokenizer: The tokenizer to use for encoding prompts.
        prompts (list): List of prompts to generate prefixes for.
        targets (list): List of target strings corresponding to the prompts.
        filtered_vocab (list): List of filtered tokens to use in the generation.
        config: Configuration parameters for the generation.

    Returns:
        list: List of generated adversarial prefixes.
    """
    gcg_config = nanogcg.GCGConfig(
        **config
    )
    generated_suffixes = []
    for prompt, target in zip(prompts, targets):
        prefix = nanogcg.run(
            model=model,
            tokenizer=tokenizer,
            messages=prompt,
            target=target,
            config=gcg_config
        )
        generated_suffixes.append(prompt + "" + prefix.best_string)
    return generated_suffixes
        