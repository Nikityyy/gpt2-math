import numpy as np
import torch
import src.utils as utils
import src.embeddings as embeddings

def compare_matmul(matrix1, matrix2):
    result1 = utils.matrix_multiply.matmul(matrix1, matrix2)
    result1 = np.array(result1)
    result2 = np.matmul(matrix1, matrix2)
    
    assert np.array_equal(result1, result2), "The results of the two matmul implementations do not match."
    print("Matrix multiplication results match!")

def compare_add_matrices(matrix1, matrix2):
    result1 = utils.add_matrices.add_matrices(matrix1, matrix2)
    result1 = np.array(result1)
    result2 = np.add(matrix1, matrix2)
    
    assert np.array_equal(result1, result2), "The results of the two add_matrices implementations do not match."
    print("Add matrices results match!")

def compare_transpose_matrix(matrix):
    result1 = utils.transpose_matrix.transpose_matrix(matrix)
    result1 = np.array(result1)
    result2 = np.transpose(matrix)
    
    assert np.array_equal(result1, result2), "The results of the two transpose implementations do not match."
    print("Transpose matrix results match!")

def compare_softmax(vector):
    result1 = utils.softmax.softmax(vector)
    result1 = np.array(result1)
    result2 = torch.nn.functional.softmax(torch.tensor(vector), dim=0).numpy()
    
    assert np.allclose(result1, result2), "The results of the two softmax implementations do not match."
    print("Softmax results match!")

def compare_masked_softmax(vector, mask):
    result1 = utils.masked_softmax.masked_softmax(vector, mask)
    result1 = np.array(result1)
    
    masked_vector = [v if m else float('-inf') for v, m in zip(vector, mask)]
    tensor_vector = torch.tensor(masked_vector)
    result2 = torch.nn.functional.softmax(tensor_vector, dim=0).numpy()
    
    assert np.allclose(result1, result2), "The results of the two masked_softmax implementations do not match."
    print("Masked softmax results match!")

def compare_layer_norm(vector):
    result1 = utils.layer_norm.layer_norm(vector)
    result1 = np.array(result1)

    layer_norm = torch.nn.LayerNorm(len(vector))
    result2 = layer_norm(torch.tensor(vector)).detach().numpy()
    
    assert np.allclose(result1, result2), "The results of the two layer_norm implementations do not match."
    print("Layer norm results match!")

def compare_token_embeddings_lookup(emb, batch_token_ids):
    result1 = embeddings.token_embeddings.token_embeddings_lookup(emb, batch_token_ids)
    result1 = np.array(result1)
 
    embedding_matrix = np.array([emb[i] for i in range(len(emb))])
    result2 = np.array([[embedding_matrix[token_id] for token_id in seq] for seq in batch_token_ids])
    
    assert np.array_equal(result1, result2), "The results of the two token_embeddings_lookup implementations do not match."
    print("Token embeddings lookup results match!")

if __name__ == "__main__":
    mat1 = [[1, 2, 3],
            [4, 5, 6]]
    
    mat2 = [[7, 8],
            [9, 10],
            [11, 12]]
    
    mat3 = [[1, 2, 3],
            [4, 5, 6]]
    
    mat4 = [[7, 8, 9],
            [10, 11, 12]]
    
    vec1 = [1.0, 2.0, 3.0, 4.0]
    
    mask = [1, 0, 1, 1]
    
    vocab_size = 10
    embedding_dim = 5
    emb = embeddings.token_embeddings.init_random_embeddings(vocab_size, embedding_dim)
    batch_token_ids = [[0, 1, 2], [3, 4, 5]]
    
    compare_matmul(mat1, mat2)
    compare_add_matrices(mat3, mat4)
    compare_transpose_matrix(mat1)
    compare_softmax(vec1)
    compare_masked_softmax(vec1, mask)
    compare_layer_norm(vec1)
    compare_token_embeddings_lookup(emb, batch_token_ids)
