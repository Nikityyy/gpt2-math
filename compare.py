import numpy as np
import torch
import src.utils as utils
import src.embeddings as embeddings
import src.layers as layers

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

def compare_positional_encoding(seq_len, embedding_dim):
    result1 = embeddings.positional_encoding.sinusoidal_positional_encoding(seq_len, embedding_dim)
    result1 = np.array(result1)

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
    pe = torch.zeros(seq_len, embedding_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    result2 = pe.numpy()

    assert np.allclose(result1, result2), "The results of the two positional_encoding implementations do not match."
    print("Positional encoding results match!")

def compare_embeddings_layer(token_embeddings, positional_encodings):
    result1 = embeddings.embeddings_layer.embeddings_layer(token_embeddings, positional_encodings)
    result1 = np.array(result1)
    
    token_embeddings_t = torch.tensor(token_embeddings, dtype=torch.float32)
    positional_encodings_t = torch.tensor(positional_encodings, dtype=torch.float32)
    seq_len = token_embeddings_t.size(1)
    pos_sliced_t = positional_encodings_t[:seq_len, :].unsqueeze(0).expand(token_embeddings_t.size(0), -1, -1)
    result2 = token_embeddings_t + pos_sliced_t
    
    assert result1.shape == result2.shape, f"Shape mismatch: {result1.shape} vs {result2.shape}"
    assert np.allclose(result1, result2.numpy()), "The results of the two embeddings_layer implementations do not match."
    
    print("Embeddings layer results match!")

def compare_linear_layer(input_vector, weight_matrix, bias_vector):
    result1 = layers.linear_layer.linear_layer(input_vector, weight_matrix, bias_vector)
    result1 = np.array(result1)

    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    weight_tensor = torch.tensor(weight_matrix, dtype=torch.float32)
    bias_tensor = torch.tensor(bias_vector, dtype=torch.float32)
    result2 = torch.matmul(input_tensor, weight_tensor) + bias_tensor
    result2 = result2.squeeze(0).numpy()

    assert np.allclose(result1, result2), "The results of the two linear_layer implementations do not match."
    print("Linear layer results match!")

def compare_scaled_dot_product_attention(queries, keys, values, mask=None):
    result1 = layers.scaled_dot_product_attention.scaled_dot_product_attention(queries, keys, values, mask)
    result1 = np.array(result1)

    result2 = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(queries, dtype=torch.float32),
        torch.tensor(keys, dtype=torch.float32),
        torch.tensor(values, dtype=torch.float32),
        attn_mask=None if mask is None else torch.tensor(mask, dtype=torch.bool)
    ).detach().numpy()

    assert np.allclose(result1, result2), "The results of the two scaled_dot_product_attention implementations do not match."
    print("Scaled dot-product attention results match!")

def compare_multi_head_attention(x, weights, mask=None):
    result1 = layers.multi_head_attention.multi_head_attention(x, weights, mask)
    result1 = np.array(result1)

    qkv_weights, Wo, d_model, num_heads, d_head = weights
    x_t = torch.tensor(x, dtype=torch.float32)
    Wo_t = torch.tensor(Wo, dtype=torch.float32)
    
    torch_head_outputs = []
    for h in range(num_heads):
        Wq_h, Wk_h, Wv_h = qkv_weights[h]
        Wq_h_t, Wk_h_t, Wv_h_t = map(lambda w: torch.tensor(w, dtype=torch.float32), (Wq_h, Wk_h, Wv_h))
        
        queries_h_t = torch.matmul(x_t, Wq_h_t)
        keys_h_t = torch.matmul(x_t, Wk_h_t)
        values_h_t = torch.matmul(x_t, Wv_h_t)
        
        head_output = torch.nn.functional.scaled_dot_product_attention(queries_h_t, keys_h_t, values_h_t, attn_mask=None)
        torch_head_outputs.append(head_output)

    concatenated_t = torch.cat(torch_head_outputs, dim=-1)
    output_t = torch.matmul(concatenated_t, Wo_t)
    result2 = output_t.numpy()

    assert np.allclose(result1, result2), "The results of the two multi_head_attention implementations do not match."
    print("Multi-head attention results match!")

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
    sequence_length = 6
    embedding_dim = 4
    emb = embeddings.token_embeddings.init_random_embeddings(vocab_size, embedding_dim)
    batch_token_ids = [[0, 1, 2], [3, 4, 5]]
    
    token_embeddings = embeddings.token_embeddings.token_embeddings_lookup(emb, batch_token_ids)
    positional_encodings = embeddings.positional_encoding.sinusoidal_positional_encoding(sequence_length, embedding_dim)
    
    weight_matrix, bias_vector = layers.linear_layer.init_random_linear(embedding_dim, embedding_dim)
    vec_input = [0.5 for _ in range(embedding_dim)]
    weights = layers.multi_head_attention.init_multi_head_attention(d_model=embedding_dim, num_heads=2)

    compare_matmul(mat1, mat2)
    compare_add_matrices(mat3, mat4)
    compare_transpose_matrix(mat1)
    compare_softmax(vec1)
    compare_masked_softmax(vec1, mask)
    compare_layer_norm(vec1)
    compare_token_embeddings_lookup(emb, batch_token_ids)
    compare_positional_encoding(sequence_length, embedding_dim)
    compare_embeddings_layer(token_embeddings, positional_encodings)
    compare_linear_layer(vec_input, weight_matrix, bias_vector)
    compare_scaled_dot_product_attention(token_embeddings, token_embeddings, token_embeddings)
    compare_multi_head_attention(token_embeddings, weights)
