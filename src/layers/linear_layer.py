import math
import random
from src.utils.matrix_multiply import matmul
from src.utils.add_matrices import add_matrices

def init_random_linear(in_dim, out_dim, num_layers=None):
    # Per GPT-2 paper, scale weights of residual layers
    # This scaling is applied to the second FFN layer and the MHA projection layer
    scale = 1.0
    if num_layers is not None:
        scale = 1 / math.sqrt(num_layers)

    weight_matrix = [[random.gauss(0, 0.02) * scale for _ in range(out_dim)] for _ in range(in_dim)]
    bias_vector = [0.0 for _ in range(out_dim)]
    return weight_matrix, bias_vector

def linear_layer(x, weight_matrix, bias_vector):
    output = matmul(x, weight_matrix)
    
    output_with_bias = add_matrices(output, bias_vector)
    
    return output_with_bias
