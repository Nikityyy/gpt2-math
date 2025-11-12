import random

def init_random_embeddings(vocab_size, embedding_dim):
    embeddings = {}
    for i in range(vocab_size):
        embeddings[i] = [random.gauss(0, 0.02) for _ in range(embedding_dim)]
    return embeddings

def token_embeddings_lookup(embeddings, batch_token_ids):
    return [[embeddings[token_id] for token_id in seq] for seq in batch_token_ids]
