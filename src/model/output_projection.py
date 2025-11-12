from src.layers.linear_layer import linear_layer
from src.utils.transpose_matrix import transpose_matrix

def output_projection(x, token_embedding_matrix):
    weight = transpose_matrix(token_embedding_matrix)
    
    vocab_size = len(token_embedding_matrix)
    bias = [0.0] * vocab_size
    
    logits = linear_layer(x, weight, bias)
    
    return logits
