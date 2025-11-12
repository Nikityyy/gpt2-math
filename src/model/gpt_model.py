
import random
import src.embeddings as embeddings
from src.utils.softmax import softmax
from src.utils.transpose_matrix import transpose_matrix
from .gpt_decoder import gpt_decoder, init_gpt_decoder
from .output_projection import output_projection

def init_gpt_model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
    weights = {
        "token_embeddings": embeddings.token_embeddings.init_random_embeddings(vocab_size, d_model),
        "positional_encodings": embeddings.positional_encoding.sinusoidal_positional_encoding(max_seq_len, d_model),
        "decoder": init_gpt_decoder(num_layers, d_model, num_heads, d_ff)
    }
    return weights

def gpt_model_forward(batch_token_ids, weights, mask=None):
    token_embedding_matrix = weights["token_embeddings"]
    token_embeds = embeddings.token_embeddings.token_embeddings_lookup(token_embedding_matrix, batch_token_ids)
    
    pos_encodings = weights["positional_encodings"]
    x = embeddings.embeddings_layer.embeddings_layer(token_embeds, pos_encodings)
    
    decoder_weights = weights["decoder"]
    decoder_output = gpt_decoder(x, decoder_weights, mask)

    projection_weight = transpose_matrix(token_embedding_matrix)
    logits = output_projection(decoder_output, projection_weight)
    
    return logits

def generate(weights, prompt_token_ids, max_new_tokens, temperature=1.0, top_k=10):
    generated_ids = list(prompt_token_ids)
    
    for _ in range(max_new_tokens):
        current_seq_len = len(generated_ids)
        
        input_ids_batch = [generated_ids]
        
        mask = [[True] * (i + 1) + [False] * (current_seq_len - 1 - i) for i in range(current_seq_len)]

        logits_batch = gpt_model_forward(input_ids_batch, weights, mask)
        
        last_token_logits = logits_batch[0][-1]
        
        if temperature > 0:
            scaled_logits = [l / temperature for l in last_token_logits]
        else:
            scaled_logits = last_token_logits

        probabilities = softmax(scaled_logits)

        if top_k > 0 and top_k < len(probabilities):
            indexed_probs = list(enumerate(probabilities))
            indexed_probs.sort(key=lambda x: x[1], reverse=True)
            top_k_indexed_probs = indexed_probs[:top_k]
            
            top_k_indices = [item[0] for item in top_k_indexed_probs]
            top_k_probs = [item[1] for item in top_k_indexed_probs]
            
            sum_top_k_probs = sum(top_k_probs)
            renormalized_probs = [p / sum_top_k_probs for p in top_k_probs]
            
            candidate_tokens = top_k_indices
            candidate_probs = renormalized_probs
        else:
            candidate_tokens = list(range(len(probabilities)))
            candidate_probs = probabilities

        r = random.random()
        cumulative_prob = 0.0
        next_token_id = -1
        for token_id, prob in zip(candidate_tokens, candidate_probs):
            cumulative_prob += prob
            if r < cumulative_prob:
                next_token_id = token_id
                break
        
        if next_token_id == -1:
            next_token_id = candidate_tokens[-1]
            
        generated_ids.append(next_token_id)
        
    return generated_ids
