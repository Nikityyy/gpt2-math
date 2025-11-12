import random

def init_random_embeddings(vocab_size, embedding_dim):
    return [[random.gauss(0, 0.02) for _ in range(embedding_dim)] for _ in range(vocab_size)]

def token_embeddings_lookup(embeddings_matrix, batch_token_ids):
    return [[embeddings_matrix[token_id] for token_id in seq] for seq in batch_token_ids]
