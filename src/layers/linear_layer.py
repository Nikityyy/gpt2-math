import random
from src.utils.matrix_multiply import matmul
from src.utils.add_matrices import add_matrices

def init_random_linear(in_dim, out_dim):
    weight_matrix = [[random.gauss(0, 0.02) for _ in range(out_dim)] for _ in range(in_dim)]
    bias_vector = [0.0 for _ in range(out_dim)]
    return weight_matrix, bias_vector

def linear_layer(input_vector, weight_matrix, bias_vector):
    output_vector = matmul([input_vector], weight_matrix)[0]
    output_vector = add_matrices([output_vector], [bias_vector])[0]
    return output_vector

