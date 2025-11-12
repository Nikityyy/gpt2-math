import math
import numpy as np
import torch
import src.utils as utils
import src.embeddings as embeddings
import src.layers as layers
import src.model as model

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
    
    input_arr = np.array(matrix)
    if input_arr.ndim == 3:
        result2 = np.transpose(input_arr, (0, 2, 1))
    else:
        result2 = np.transpose(input_arr)
    
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

def compare_layer_norm(input_data):
    temp = input_data
    while isinstance(temp, list) and temp and isinstance(temp[0], list):
        temp = temp[0]
    
    last_dim = len(temp)

    gamma, beta = utils.layer_norm.init_layer_norm(last_dim)
    result1 = utils.layer_norm.layer_norm(input_data, gamma, beta)
    result1 = np.array(result1)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    normalized_shape = input_tensor.shape[-1]
    torch_ln = torch.nn.LayerNorm(normalized_shape)

    torch_ln.weight.data.fill_(1.0)
    torch_ln.bias.data.fill_(0.0)
    result2 = torch_ln(input_tensor).detach().numpy()

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

def compare_linear_layer(input_tensor_3d, weight_matrix, bias_vector):
    result1 = layers.linear_layer.linear_layer(input_tensor_3d, weight_matrix, bias_vector)
    result1 = np.array(result1)

    in_dim = len(weight_matrix)
    out_dim = len(bias_vector)
    
    input_t = torch.tensor(input_tensor_3d, dtype=torch.float32)
    
    torch_linear = torch.nn.Linear(in_dim, out_dim)
    
    torch_linear.weight.data = torch.tensor(np.array(weight_matrix).T, dtype=torch.float32)
    torch_linear.bias.data = torch.tensor(bias_vector, dtype=torch.float32)
    
    result2_t = torch_linear(input_t)
    result2 = result2_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the two linear_layer implementations do not match."
    print("Linear layer results match!")

def compare_scaled_dot_product_attention(queries, keys, values, mask=None):
    result1 = layers.scaled_dot_product_attention.scaled_dot_product_attention(queries, keys, values, mask)
    result1 = np.array(result1)

    queries_t = torch.tensor(queries, dtype=torch.float32)
    keys_t = torch.tensor(keys, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    
    d_k = keys_t.size(-1)
    scores_t = torch.matmul(queries_t, keys_t.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        torch_mask = ~torch.tensor(mask, dtype=torch.bool)
        scores_t = scores_t.masked_fill(torch_mask, -float('inf'))
    
    attention_weights_t = torch.nn.functional.softmax(scores_t, dim=-1)
    
    output_t = torch.matmul(attention_weights_t, values_t)
    result2 = output_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the two scaled_dot_product_attention implementations do not match."
    print("Scaled dot-product attention results match!")

def compare_multi_head_attention(x, weights, mask=None):
    result1 = layers.multi_head_attention.multi_head_attention(x, weights, mask)
    result1 = np.array(result1)

    qkv_weights, (Wo, bo), d_model, num_heads, d_head = weights
    
    x_t = torch.tensor(x, dtype=torch.float32)
    Wo_t = torch.tensor(Wo, dtype=torch.float32)
    bo_t = torch.tensor(bo, dtype=torch.float32)
    
    torch_mask = None
    if mask is not None:
        torch_mask = ~torch.tensor(mask, dtype=torch.bool)
    
    torch_head_outputs = []
    for h in range(num_heads):
        (Wq_h, bq_h), (Wk_h, bk_h), (Wv_h, bv_h) = qkv_weights[h]
        
        Wq_h_t, Wk_h_t, Wv_h_t = map(lambda w: torch.tensor(w, dtype=torch.float32), (Wq_h, Wk_h, Wv_h))
        bq_h_t, bk_h_t, bv_h_t = map(lambda b: torch.tensor(b, dtype=torch.float32), (bq_h, bk_h, bv_h))
        
        queries_h_t = torch.matmul(x_t, Wq_h_t) + bq_h_t
        keys_h_t = torch.matmul(x_t, Wk_h_t) + bk_h_t
        values_h_t = torch.matmul(x_t, Wv_h_t) + bv_h_t
        
        d_k = queries_h_t.size(-1)
        scores = torch.matmul(queries_h_t, keys_h_t.transpose(-2, -1)) / np.sqrt(d_k)
        if torch_mask is not None:
            scores = scores.masked_fill(torch_mask, -float('inf'))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        head_output = torch.matmul(attention_weights, values_h_t)
        
        torch_head_outputs.append(head_output)

    concatenated_t = torch.cat(torch_head_outputs, dim=-1)
    output_t = torch.matmul(concatenated_t, Wo_t) + bo_t
    result2 = output_t.numpy()

    assert np.allclose(result1, result2), "The results of the two multi_head_attention implementations do not match."
    print("Multi-head attention results match!")

def compare_feed_forward(x, weights):
    result1 = layers.feed_forward.feed_forward(x, weights)
    result1 = np.array(result1)

    (W1, b1), (W2, b2) = weights
    x_t = torch.tensor(x, dtype=torch.float32)

    linear1 = torch.nn.Linear(len(W1), len(W1[0]))
    linear1.weight.data = torch.tensor(np.array(W1).T, dtype=torch.float32)
    linear1.bias.data = torch.tensor(b1, dtype=torch.float32)

    gelu_activation = torch.nn.GELU(approximate='tanh')

    linear2 = torch.nn.Linear(len(W2), len(W2[0]))
    linear2.weight.data = torch.tensor(np.array(W2).T, dtype=torch.float32)
    linear2.bias.data = torch.tensor(b2, dtype=torch.float32)
    
    result2_t = linear2(gelu_activation(linear1(x_t)))
    result2 = result2_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the feed_forward implementations do not match."
    print("Feed-forward network results match!")

def compare_transformer_block(x, weights, mask=None):
    result1 = layers.transformer_block.transformer_block(x, weights, mask)
    result1 = np.array(result1)

    mha_weights, ffn_weights, ln1_weights, ln2_weights = weights
    
    qkv_weights, (Wo, bo), d_model, num_heads, d_head = mha_weights
    Wo_t = torch.tensor(Wo, dtype=torch.float32)
    bo_t = torch.tensor(bo, dtype=torch.float32)

    (W1, b1), (W2, b2) = ffn_weights
    
    x_t = torch.tensor(x, dtype=torch.float32)

    torch_ln1 = torch.nn.LayerNorm(d_model)
    torch_ln1.weight.data.fill_(1.0)
    torch_ln1.bias.data.fill_(0.0)
    
    torch_ln2 = torch.nn.LayerNorm(d_model)
    torch_ln2.weight.data.fill_(1.0)
    torch_ln2.bias.data.fill_(0.0)
    
    torch_ffn = torch.nn.Sequential(
        torch.nn.Linear(d_model, len(W1[0])),
        torch.nn.GELU(approximate='tanh'),
        torch.nn.Linear(len(W2), len(W2[0]))
    )
    torch_ffn[0].weight.data = torch.tensor(np.array(W1).T, dtype=torch.float32)
    torch_ffn[0].bias.data = torch.tensor(b1, dtype=torch.float32)
    torch_ffn[2].weight.data = torch.tensor(np.array(W2).T, dtype=torch.float32)
    torch_ffn[2].bias.data = torch.tensor(b2, dtype=torch.float32)
    torch_ffn.eval()

    normed_x_t = torch_ln1(x_t)
    
    torch_mask = ~torch.tensor(mask, dtype=torch.bool) if mask is not None else None
    
    torch_head_outputs = []
    for h in range(num_heads):
        (Wq_h, bq_h), (Wk_h, bk_h), (Wv_h, bv_h) = qkv_weights[h]
        
        Wq_h_t, Wk_h_t, Wv_h_t = map(lambda w: torch.tensor(w, dtype=torch.float32), (Wq_h, Wk_h, Wv_h))
        bq_h_t, bk_h_t, bv_h_t = map(lambda b: torch.tensor(b, dtype=torch.float32), (bq_h, bk_h, bv_h))
        
        queries_h_t = torch.matmul(normed_x_t, Wq_h_t) + bq_h_t
        keys_h_t = torch.matmul(normed_x_t, Wk_h_t) + bk_h_t
        values_h_t = torch.matmul(normed_x_t, Wv_h_t) + bv_h_t
        
        d_k = queries_h_t.size(-1)
        scores = torch.matmul(queries_h_t, keys_h_t.transpose(-2, -1)) / np.sqrt(d_k)
        if torch_mask is not None:
            scores = scores.masked_fill(torch_mask, -float('inf'))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        head_output = torch.matmul(attention_weights, values_h_t)
        torch_head_outputs.append(head_output)

    concatenated_t = torch.cat(torch_head_outputs, dim=-1)
    attn_output_t = torch.matmul(concatenated_t, Wo_t) + bo_t
    
    residual1_t = x_t + attn_output_t
    
    normed_residual1_t = torch_ln2(residual1_t)

    ffn_output_t = torch_ffn(normed_residual1_t)
    
    result2_t = residual1_t + ffn_output_t
    result2 = result2_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the two transformer_block implementations do not match."
    print("Transformer block results match!")

def compare_gpt_decoder(x, weights, mask=None):
    result1 = model.gpt_decoder.gpt_decoder(x, weights, mask)
    result1 = np.array(result1)

    block_weights_list, final_ln_weights = weights
    num_layers = len(block_weights_list)
    
    _, _, d_model, num_heads, _ = block_weights_list[0][0]
    
    hidden_state_t = torch.tensor(x, dtype=torch.float32)
    torch_mask = ~torch.tensor(mask, dtype=torch.bool) if mask is not None else None

    for i in range(num_layers):
        mha_weights, ffn_weights, ln1_weights, ln2_weights = block_weights_list[i]
        qkv_weights, (Wo, bo), _, _, _ = mha_weights
        (W1, b1), (W2, b2) = ffn_weights

        torch_ln1 = torch.nn.LayerNorm(d_model)
        torch_ln1.weight.data.fill_(1.0)
        torch_ln1.bias.data.fill_(0.0)
        normed_x_t = torch_ln1(hidden_state_t)
        
        torch_head_outputs = []
        for h in range(num_heads):
            (Wq_h, bq_h), (Wk_h, bk_h), (Wv_h, bv_h) = qkv_weights[h]
            Wq_h_t, Wk_h_t, Wv_h_t = map(lambda w: torch.tensor(w, dtype=torch.float32), (Wq_h, Wk_h, Wv_h))
            bq_h_t, bk_h_t, bv_h_t = map(lambda b: torch.tensor(b, dtype=torch.float32), (bq_h, bk_h, bv_h))
            
            queries_h_t = torch.matmul(normed_x_t, Wq_h_t) + bq_h_t
            keys_h_t = torch.matmul(normed_x_t, Wk_h_t) + bk_h_t
            values_h_t = torch.matmul(normed_x_t, Wv_h_t) + bv_h_t
            
            d_k = queries_h_t.size(-1)
            scores = torch.matmul(queries_h_t, keys_h_t.transpose(-2, -1)) / np.sqrt(d_k)
            if torch_mask is not None:
                scores = scores.masked_fill(torch_mask, -float('inf'))
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)
            head_output = torch.matmul(attention_weights, values_h_t)
            torch_head_outputs.append(head_output)

        concatenated_t = torch.cat(torch_head_outputs, dim=-1)
        attn_output_t = torch.matmul(concatenated_t, torch.tensor(Wo, dtype=torch.float32)) + torch.tensor(bo, dtype=torch.float32)
        
        residual1_t = hidden_state_t + attn_output_t
        
        torch_ln2 = torch.nn.LayerNorm(d_model)
        torch_ln2.weight.data.fill_(1.0)
        torch_ln2.bias.data.fill_(0.0)
        normed_residual1_t = torch_ln2(residual1_t)
        
        torch_ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, len(W1[0])), torch.nn.GELU(approximate='tanh'), torch.nn.Linear(len(W2), len(W2[0]))
        )
        torch_ffn[0].weight.data = torch.tensor(np.array(W1).T, dtype=torch.float32)
        torch_ffn[0].bias.data = torch.tensor(b1, dtype=torch.float32)
        torch_ffn[2].weight.data = torch.tensor(np.array(W2).T, dtype=torch.float32)
        torch_ffn[2].bias.data = torch.tensor(b2, dtype=torch.float32)
        ffn_output_t = torch_ffn(normed_residual1_t)
        
        hidden_state_t = residual1_t + ffn_output_t

    final_gamma, final_beta = final_ln_weights
    torch_final_ln = torch.nn.LayerNorm(d_model)
    torch_final_ln.weight.data = torch.tensor(final_gamma, dtype=torch.float32)
    torch_final_ln.bias.data = torch.tensor(final_beta, dtype=torch.float32)
    
    result2_t = torch_final_ln(hidden_state_t)
    result2 = result2_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the gpt_decoder implementations do not match."
    print("GPT decoder results match!")

def compare_output_projection(x, token_embedding_matrix):
    result1 = model.output_projection.output_projection(x, token_embedding_matrix)
    result1 = np.array(result1)

    x_t = torch.tensor(x, dtype=torch.float32)
    embedding_matrix_t = torch.tensor(token_embedding_matrix, dtype=torch.float32)

    vocab_size, d_model = embedding_matrix_t.shape

    torch_linear_projection = torch.nn.Linear(d_model, vocab_size, bias=False)
    
    torch_linear_projection.weight.data = embedding_matrix_t
    
    result2_t = torch_linear_projection(x_t)
    result2 = result2_t.detach().numpy()

    assert np.allclose(result1, result2), "The results of the output_projection implementations do not match."
    print("Output projection results match!")

if __name__ == "__main__":
    vocab_size = 10
    batch_size = 2
    sequence_length = 3
    embedding_dim = 4
    num_layers = 2
    num_heads = 2
    d_ff = embedding_dim * 4

    emb = embeddings.token_embeddings.init_random_embeddings(vocab_size, embedding_dim)
    
    batch_token_ids = np.random.randint(0, vocab_size, size=(batch_size, sequence_length)).tolist()
    
    token_embeddings_val = embeddings.token_embeddings.token_embeddings_lookup(emb, batch_token_ids)
    
    positional_encodings = embeddings.positional_encoding.sinusoidal_positional_encoding(sequence_length * 2, embedding_dim)
    
    attention_mask = [[True] * (i + 1) + [False] * (sequence_length - 1 - i) for i in range(sequence_length)]

    weight_matrix, bias_vector = layers.linear_layer.init_random_linear(embedding_dim, embedding_dim)
    mha_weights_for_test = layers.multi_head_attention.init_multi_head_attention(d_model=embedding_dim, num_heads=num_heads)
    ffn_weights_for_test = layers.feed_forward.init_feed_forward(d_model=embedding_dim, d_ff=d_ff)
    block_weights_for_test = layers.transformer_block.init_transformer_block(embedding_dim, num_heads, d_ff)

    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    vec1 = [1.0, 2.0, 3.0, 4.0]
    mask_1d = [True, False, True, True]
    
    block_weights_for_test = layers.transformer_block.init_transformer_block(embedding_dim, num_heads, d_ff)
    
    # Add new initialization for the decoder
    decoder_weights_for_test = model.gpt_decoder.init_gpt_decoder(num_layers, embedding_dim, num_heads, d_ff)
    
    # Pre-compute embeddings for input to layers
    embeddings_output = embeddings.embeddings_layer.embeddings_layer(token_embeddings_val, positional_encodings)
    
    print("--- Testing Utils ---")
    compare_matmul(mat1, mat2)
    compare_add_matrices(mat1, mat2)
    compare_transpose_matrix(mat1)
    compare_softmax(vec1)
    compare_masked_softmax(vec1, mask_1d)
    compare_layer_norm(vec1) # Test 1D
    compare_layer_norm(token_embeddings_val) # Test 3D

    print("\n--- Testing Embeddings ---")
    compare_token_embeddings_lookup(emb, batch_token_ids)
    compare_positional_encoding(sequence_length, embedding_dim)
    compare_embeddings_layer(token_embeddings_val, positional_encodings)
    
    print("\n--- Testing Layers ---")
    compare_linear_layer(token_embeddings_val, weight_matrix, bias_vector)
    compare_scaled_dot_product_attention(token_embeddings_val, token_embeddings_val, token_embeddings_val, mask=attention_mask)
    compare_multi_head_attention(token_embeddings_val, mha_weights_for_test, mask=attention_mask)
    compare_feed_forward(token_embeddings_val, ffn_weights_for_test)
    
    print("\n--- Testing Full Transformer Block ---")
    compare_transformer_block(token_embeddings_val, block_weights_for_test, mask=attention_mask)

    print("\n--- Testing GPT Decoder ---")
    decoder_output = model.gpt_decoder.gpt_decoder(embeddings_output, decoder_weights_for_test, mask=attention_mask)
    compare_gpt_decoder(embeddings_output, decoder_weights_for_test, mask=attention_mask)

    print("\n--- Testing Output Projection ---")
    token_embedding_matrix = [emb[i] for i in range(vocab_size)]
    compare_output_projection(decoder_output, token_embedding_matrix)
