import os
import random
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATASET_ROOT = '~/.cache/huggingface/hub/'

class SEFPrepareHelper:
    def __init__(self, config):
        self.config = config
        self.qa_csv_path = config.get('qa_csv_path', './sef_qa_samples.csv')
        self.n_samples = config.get('n_samples', 200)
        self.resample = config.get('resample', False)
        self.qa_datasets = config.get('qa_datasets', ['truthful_qa', 'squad'])
        self.embedding_model_name = config.get('embedding_model_name', 'Qwen/Qwen3-Embedding-4B')
        self.continuation_word_limit = config.get('continuation_word_limit', 20)
        random.seed(config.get('random_seed', 42))

    def get_qa_samples(self):
        if os.path.exists(self.qa_csv_path) and not self.resample:
            df = pd.read_csv(self.qa_csv_path)
            return df['question'].tolist()
        
        # Calculate samples per dataset
        samples_per_dataset = self.n_samples // len(self.qa_datasets)
        remaining_samples = self.n_samples % len(self.qa_datasets)
        
        all_questions = []
        
        for i, dataset_name in enumerate(self.qa_datasets):
            # Add one extra sample for first few datasets if there's remainder
            current_samples = samples_per_dataset + (1 if i < remaining_samples else 0)
            questions = self._load_dataset_questions(dataset_name, current_samples)
            all_questions.extend(questions)
        
        # Shuffle all questions
        random.shuffle(all_questions)
        os.makedirs(os.path.dirname(self.qa_csv_path), exist_ok=True)
        pd.DataFrame({'question': all_questions}).to_csv(self.qa_csv_path, index=False)
        return all_questions
    
    def _load_dataset_questions(self, dataset_name, num_samples):
        """Load questions from a specific dataset."""
        questions = []
        
        try:
            if dataset_name == 'truthful_qa':
                ds = load_dataset('truthful_qa', 'generation', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'squad':
                ds = load_dataset('squad', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'metaeval/reclor':
                ds = load_dataset('metaeval/reclor', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]

            elif dataset_name == 'ucinlp/drop':
                ds = load_dataset('ucinlp/drop', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'hendrycks/ethics':
                # Ethics dataset has multiple subsets, use 'commonsense' as default
                ds = load_dataset('hendrycks/ethics', 'commonsense', split='test', streaming=True, trust_remote_code=True)
                questions = [f"Is this statement ethical: {x['input']}" for x in ds]
                
            elif dataset_name == 'allenai/social_i_qa':
                ds = load_dataset('allenai/social_i_qa', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'qiaojin/PubMedQA':
                ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'sciq':
                ds = load_dataset('sciq', split='validation', streaming=True, trust_remote_code=True)
                questions = [x['question'] for x in ds]
                
            elif dataset_name == 'openai_humaneval':
                ds = load_dataset('openai_humaneval', split='test', streaming=True, trust_remote_code=True)
                questions = []
                for x in ds:
                    # Extract first n words from the docstring/prompt
                    prompt_text = x['prompt']
                    words = prompt_text.split()[:self.continuation_word_limit]
                    truncated_text = ' '.join(words)
                    question = f"Complete the following code: {truncated_text}"
                    questions.append(question)
                    
            elif dataset_name == 'pkoerner/cam_stories':
                ds = load_dataset('pkoerner/cam_stories', split='val', streaming=True, trust_remote_code=True)
                questions = []
                for x in ds:
                    # Extract first n words from the story
                    story_text = x['story']
                    words = story_text.split()[:self.continuation_word_limit]
                    truncated_text = ' '.join(words)
                    question = f"Continue the following story: {truncated_text}"
                    questions.append(question)
                    
            else:
                print(f"Warning: Unknown dataset {dataset_name}, skipping...")
                return []
                
            # Remove duplicates and sample
            questions = list(set(questions))
            if len(questions) >= num_samples:
                sampled_questions = random.sample(questions, num_samples)
            else:
                print(f"Warning: Dataset {dataset_name} has only {len(questions)} samples, less than requested {num_samples}")
                sampled_questions = questions
                
            return sampled_questions
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []

    def get_sentence_embedding_model(self):
        return SentenceTransformer(self.embedding_model_name)
