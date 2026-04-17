import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

class VectorEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the Vector Engine by downloading/loading a lightweight open-source 
        Transformer model from HuggingFace to memory.
        """
        # Load the vocabulary (tokenizer) and the raw weights (model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def create_chunks(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        """
        A minimalist custom chunking algorithm.
        Splits a large document into overlapping windows to preserve context.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Tokenizes the input text, runs it through the neural network,
        and manually extracts the embedding using Mean Pooling.
        """
        # 1. Tokenize the text into numbers the GPU/CPU can understand
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", # Return PyTorch tensors
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 2. Forward Pass: Run the neural network without calculating gradients (inference only)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 3. Mean Pooling Architecture implementation (Core of semantic compression)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Expand the attention mask to match the dimensions of the token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Compute the sum of embeddings and sum of mask, handling division safely
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate the final single sentence embedding
        embedding = sum_embeddings / sum_mask
        
        # 4. Conversion to native NumPy array for raw mathematical calculations
        return embedding.squeeze().numpy()

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates the semantic similarity between two vectors using pure math.
        Formula: (A dot B) / (||A|| * ||B||)
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        # Edge case handler for 0-magnitude vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))
