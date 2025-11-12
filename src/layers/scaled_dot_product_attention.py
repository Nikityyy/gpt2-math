import math
from src.utils.matrix_multiply import matmul
from src.utils.transpose_matrix import transpose_matrix
from src.utils.masked_softmax import masked_softmax
from src.utils.softmax import softmax

def scaled_dot_product_attention(queries, keys, values, mask=None):
    if not keys or not keys[0] or not keys[0][0]:
         raise ValueError("Invalid shape for keys in scaled_dot_product_attention")
    dk = len(keys[0][0])

    scores = matmul(queries, transpose_matrix(keys))
    # Scale scores
    scale_factor = math.sqrt(dk)
    scaled_scores = [[[score / scale_factor for score in row] for row in matrix] for matrix in scores]
    
    if mask is not None:
        attention_weights = masked_softmax(scaled_scores, mask)
    else:
        attention_weights = softmax(scaled_scores)
    
    output = matmul(attention_weights, values)
    return output
