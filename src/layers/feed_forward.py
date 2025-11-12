from .linear_layer import init_random_linear, linear_layer
from src.utils.gelu import gelu

def init_feed_forward(d_model, d_ff):
    W1, b1 = init_random_linear(d_model, d_ff)
    W2, b2 = init_random_linear(d_ff, d_model)
    return [(W1, b1), (W2, b2)]

def feed_forward(x, weights):
    (W1, b1), (W2, b2) = weights
    
    # First linear transformation
    hidden = linear_layer(x, W1, b1)
    
    # GELU activation
    activated = gelu(hidden)
    
    # Second linear transformation
    output = linear_layer(activated, W2, b2)
    
    return output
