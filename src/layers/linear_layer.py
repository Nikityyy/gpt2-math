import random
from src.utils.matrix_multiply import matmul
from src.utils.add_matrices import add_matrices

def init_random_linear(in_dim, out_dim):
    weight_matrix = [[random.gauss(0, 0.02) for _ in range(out_dim)] for _ in range(in_dim)]
    bias_vector = [0.0 for _ in range(out_dim)]
    return weight_matrix, bias_vector

def linear_layer(x, weight_matrix, bias_vector):
    output = matmul(x, weight_matrix)
    
    output_with_bias = add_matrices(output, bias_vector)
    
    return output_with_bias
