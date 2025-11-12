from .multi_head_attention import init_multi_head_attention, multi_head_attention
from .feed_forward import init_feed_forward, feed_forward
from src.utils.layer_norm import init_layer_norm, layer_norm
from src.utils.add_matrices import add_matrices

def init_transformer_block(d_model, num_heads, d_ff, num_layers=None):
    mha_weights = init_multi_head_attention(d_model, num_heads, num_layers=num_layers)
    ffn_weights = init_feed_forward(d_model, d_ff, num_layers=num_layers)
    ln1_weights = init_layer_norm(d_model)
    ln2_weights = init_layer_norm(d_model)
    return [mha_weights, ffn_weights, ln1_weights, ln2_weights]

def transformer_block(x, weights, mask=None):
    mha_weights, ffn_weights, ln1_weights, ln2_weights = weights
    
    # --- First Sub-layer: Multi-Head Attention with Pre-LN and Residual ---
    
    # 1. Pre-Layer Normalization
    normed_x = layer_norm(x, ln1_weights[0], ln1_weights[1])
    
    # 2. Multi-Head Attention
    attn_output = multi_head_attention(normed_x, mha_weights, mask)
    
    # 3. Residual Connection
    residual1 = add_matrices(x, attn_output)
    
    # --- Second Sub-layer: Feed-Forward Network with Pre-LN and Residual ---
    
    # 4. Pre-Layer Normalization
    normed_residual1 = layer_norm(residual1, ln2_weights[0], ln2_weights[1])
    
    # 5. Feed-Forward Network
    ffn_output = feed_forward(normed_residual1, ffn_weights)
    
    # 6. Residual Connection
    output = add_matrices(residual1, ffn_output)
    
    return output
