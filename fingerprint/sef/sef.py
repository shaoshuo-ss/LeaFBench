from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.sef.sef_prepare_helper import SEFPrepareHelper
from fingerprint.sef.sef_fingerprint_helper import SEFFingerprintHelper

class SEFFingerprint(LLMFingerprintInterface):
    """
    SEF (Sentence Embedding Fingerprint) Fingerprint implementation.
    """

    def __init__(self, config=None, accelerator=None):
        super().__init__(config, accelerator)

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods: sample QA questions and load sentence embedding model.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        self.helper = SEFPrepareHelper(self.config)
        self.qa_questions = self.helper.get_qa_samples()
        self.embedding_model = self.helper.get_sentence_embedding_model().to(self.accelerator.device)
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given model.

        Args:
            model: The model to extract the fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        if not hasattr(self, 'qa_questions') or not hasattr(self, 'embedding_model'):
            raise RuntimeError("SEF fingerprint is not prepared. Call prepare() first.")
        
        fingerprint_helper = SEFFingerprintHelper(self.config, self.accelerator)
        
        # Generate responses for all QA questions
        responses = fingerprint_helper.generate_responses_batch(model, self.qa_questions)
        
        # Get sentence embeddings for all responses
        embeddings = fingerprint_helper.get_sentence_embeddings_batch(self.embedding_model, responses)
        
        # Aggregate embeddings based on configuration
        fingerprint = fingerprint_helper.aggregate_embeddings(embeddings)
        
        return fingerprint
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints in [0,1] range.
        """
        fingerprint_helper = SEFFingerprintHelper(self.config, self.accelerator)
        
        # Get fingerprints for both models
        fingerprint1 = base_model.get_fingerprint()
        fingerprint2 = testing_model.get_fingerprint()
        
        # Compute cosine similarity and scale to [0,1]
        similarity_score = fingerprint_helper.compute_similarity(fingerprint1, fingerprint2)
        
        return similarity_score