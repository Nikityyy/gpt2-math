from .linear_layer import init_random_linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from src.utils.matrix_multiply import matmul

def init_multi_head_attention(d_model, num_heads):
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    
    d_head = d_model // num_heads
    
    # Initialize weights for each head
    qkv_weights = []
    for _ in range(num_heads):
        Wq, _ = init_random_linear(d_model, d_head)
        Wk, _ = init_random_linear(d_model, d_head)
        Wv, _ = init_random_linear(d_model, d_head)
        qkv_weights.append((Wq, Wk, Wv))
        
    # Output linear layer weights
    Wo, _ = init_random_linear(d_model, d_model)
    
    return qkv_weights, Wo, d_model, num_heads, d_head

def multi_head_attention(x, weights, mask=None):
    # Perform multi-head attention
    qkv_weights, Wo, d_model, num_heads, d_head = weights
    batch_size, seq_len, _ = len(x), len(x[0]), len(x[0][0])
    
    all_head_outputs = []
    
    for head_idx in range(num_heads):
        Wq, Wk, Wv = qkv_weights[head_idx]
        
        # Compute Q, K, V
        Q = matmul(x, Wq)
        K = matmul(x, Wk)
        V = matmul(x, Wv)
        
        # Scaled dot-product attention
        head_output = scaled_dot_product_attention(Q, K, V, mask)
        all_head_outputs.append(head_output)

    # Concatenate all head outputs
    concatenated_output = []
    for b in range(batch_size):
        batch_item = []
        for t in range(seq_len):
            # For this token, gather the results from all heads
            token_vector = []
            for head_output in all_head_outputs:
                token_vector.extend(head_output[b][t])
            batch_item.append(token_vector)
        concatenated_output.append(batch_item)

    # Final linear layer
    output = matmul(concatenated_output, Wo)
    
    return output
