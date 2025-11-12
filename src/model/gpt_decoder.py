from src.layers.transformer_block import init_transformer_block, transformer_block


def init_gpt_decoder(num_layers, d_model, num_heads, d_ff):
    block_weights = [
        init_transformer_block(d_model, num_heads, d_ff, num_layers=num_layers) for _ in range(num_layers)
    ]
    return block_weights

def gpt_decoder(x, weights, mask=None):
    block_weights = weights
    
    hidden_state = x
    for current_block_weights in block_weights:
        hidden_state = transformer_block(hidden_state, current_block_weights, mask)
        
    return hidden_state
