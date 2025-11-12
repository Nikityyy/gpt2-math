from .linear_layer import init_random_linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from src.utils.matrix_multiply import matmul
from .linear_layer import linear_layer

def init_multi_head_attention(d_model, num_heads):
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    
    d_head = d_model // num_heads
    
    # Initialize weights for each head
    qkv_weights = []
    for _ in range(num_heads):
        Wq, bq = init_random_linear(d_model, d_head)
        Wk, bk = init_random_linear(d_model, d_head)
        Wv, bv = init_random_linear(d_model, d_head)
        qkv_weights.append(((Wq, bq), (Wk, bk), (Wv, bv)))
        
    # Output linear layer weights
    Wo, bo = init_random_linear(d_model, d_model)
    
    return qkv_weights, (Wo, bo), d_model, num_heads, d_head

def multi_head_attention(x, weights, mask=None):
    # Perform multi-head attention
    qkv_weights, (Wo, bo), d_model, num_heads, d_head = weights
    batch_size, seq_len, _ = len(x), len(x[0]), len(x[0][0])
    
    all_head_outputs = []
    
    for head_idx in range(num_heads):
        (Wq, bq), (Wk, bk), (Wv, bv) = qkv_weights[head_idx]
        
        # Compute Q, K, V
        Q = linear_layer(x, Wq, bq)
        K = linear_layer(x, Wk, bk)
        V = linear_layer(x, Wv, bv)
        
        # Scaled dot-product attention
        head_output = scaled_dot_product_attention(Q, K, V, mask)
        all_head_outputs.append(head_output)

    # Concatenate all head outputs
    concatenated_output = [
        [
            sum((all_head_outputs[h][b][t] for h in range(num_heads)), [])
            for t in range(seq_len)
        ]
        for b in range(batch_size)
    ]

    # Final linear layer
    output = linear_layer(concatenated_output, Wo, bo)
    
    return output
