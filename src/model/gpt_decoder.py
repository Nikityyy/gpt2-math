from src.layers.transformer_block import init_transformer_block, transformer_block
from src.utils.layer_norm import init_layer_norm, layer_norm


def init_gpt_decoder(num_layers, d_model, num_heads, d_ff):
    block_weights = [
        init_transformer_block(d_model, num_heads, d_ff) for _ in range(num_layers)
    ]
    ln_weights = init_layer_norm(d_model)
    return block_weights, ln_weights

def gpt_decoder(x, weights, mask=None):
    block_weights, final_ln_weights = weights
    
    hidden_state = x
    for current_block_weights in block_weights:
        hidden_state = transformer_block(hidden_state, current_block_weights, mask)
        
    final_gamma, final_beta = final_ln_weights
    output = layer_norm(hidden_state, final_gamma, final_beta)
    
    return output
