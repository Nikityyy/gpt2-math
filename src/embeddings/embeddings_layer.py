from src.utils.add_matrices import add_matrices

def embeddings_layer(token_embeddings, positional_encodings):
    if not token_embeddings:
        return []
    
    seq_len = len(token_embeddings[0])
    if any(len(seq) != seq_len for seq in token_embeddings):
        raise ValueError("All sequences in batch must have equal lengths")
    
    sliced_pos = positional_encodings[:seq_len]
    
    combined_embeddings = []
    for batch_emb in token_embeddings:
        combined_embeddings.append(add_matrices(batch_emb, sliced_pos))
    
    return combined_embeddings
