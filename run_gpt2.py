import tiktoken
from transformers import GPT2LMHeadModel

from src.model.gpt_model import init_gpt_model, generate

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

def load_and_map_gpt2_weights():
    print("Loading pre-trained GPT-2 model from Hugging Face...")
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    model_hf.eval()
    
    pretrained_weights = model_hf.state_dict()
    config = model_hf.config

    vocab_size = config.vocab_size
    d_model = config.n_embd
    num_layers = config.n_layer
    num_heads = config.n_head
    max_seq_len = config.n_positions
    d_ff = d_model * 4  # Standard GPT-2 configuration
    d_head = d_model // num_heads

    print("Initializing our custom model structure...")
    my_weights = init_gpt_model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)

    print("Mapping pre-trained weights to our structure...")
    
    # 1. Token and Positional Embeddings
    my_weights["token_embeddings"] = tensor_to_list(pretrained_weights["transformer.wte.weight"])
    my_weights["positional_encodings"] = tensor_to_list(pretrained_weights["transformer.wpe.weight"])

    # 2. Final Layer Normalization
    my_weights["final_layer_norm"] = (
        tensor_to_list(pretrained_weights["transformer.ln_f.weight"]),
        tensor_to_list(pretrained_weights["transformer.ln_f.bias"])
    )

    # 3. Transformer Blocks (Decoder Layers)
    for i in range(num_layers):
        # LayerNorm 1 weights
        my_weights["decoder"][i][2] = (
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.ln_1.weight"]),
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.ln_1.bias"])
        )
        # LayerNorm 2 weights
        my_weights["decoder"][i][3] = (
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.ln_2.weight"]),
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.ln_2.bias"])
        )

        # Feed-Forward Network weights
        my_weights["decoder"][i][1][0] = ( # First FFN linear layer
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.mlp.c_fc.weight"]),
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.mlp.c_fc.bias"])
        )
        my_weights["decoder"][i][1][1] = ( # Second FFN linear layer
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.mlp.c_proj.weight"]),
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.mlp.c_proj.bias"])
        )
        
        # Multi-Head Attention weights
        # HF combines Q, K, V into one large matrix. We need to split it.
        qkv_weights = pretrained_weights[f"transformer.h.{i}.attn.c_attn.weight"]
        qkv_bias = pretrained_weights[f"transformer.h.{i}.attn.c_attn.bias"]
        
        # Split the combined weights and biases for Q, K, V
        qkv_weights_list = tensor_to_list(qkv_weights)
        qkv_bias_list = tensor_to_list(qkv_bias)
        Wq_combined = [row[0:d_model] for row in qkv_weights_list]
        Wk_combined = [row[d_model:2*d_model] for row in qkv_weights_list]
        Wv_combined = [row[2*d_model:] for row in qkv_weights_list]
        bq_combined = qkv_bias_list[0:d_model]
        bk_combined = qkv_bias_list[d_model:2*d_model]
        bv_combined = qkv_bias_list[2*d_model:]

        # Split Q, K, V further for each head and assign
        for h in range(num_heads):
            # Weights for head h
            Wq = [row[h*d_head : (h+1)*d_head] for row in Wq_combined]
            Wk = [row[h*d_head : (h+1)*d_head] for row in Wk_combined]
            Wv = [row[h*d_head : (h+1)*d_head] for row in Wv_combined]

            # Biases for head h
            bq = bq_combined[h*d_head : (h+1)*d_head]
            bk = bk_combined[h*d_head : (h+1)*d_head]
            bv = bv_combined[h*d_head : (h+1)*d_head]
            
            my_weights["decoder"][i][0][0][h] = ((Wq, bq), (Wk, bk), (Wv, bv))

        # Attention output projection layer
        my_weights["decoder"][i][0][1] = (
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.attn.c_proj.weight"].T),
            tensor_to_list(pretrained_weights[f"transformer.h.{i}.attn.c_proj.bias"])
        )
        
    print("Weight mapping complete!")
    return my_weights

if __name__ == "__main__":
    gpt2_weights = load_and_map_gpt2_weights()

    enc = tiktoken.get_encoding("gpt2")
    
    prompt_text = "Once upon a time"
    prompt_ids = enc.encode(prompt_text)
    
    print("-" * 50)
    print(f"Prompt: '{prompt_text}'")
    print(f"Encoded IDs: {prompt_ids}")
    print("Generating text with loaded GPT-2 weights...")

    generated_ids = generate(
        weights=gpt2_weights,
        prompt_token_ids=prompt_ids,
        max_new_tokens=5,
        temperature=0.7,
        top_k=40
    )
    
    generated_text = enc.decode(generated_ids)
    
    print("-" * 50)
    print(f"Generated IDs: {generated_ids}")
    print(f"Generated Text: '{generated_text}'")
    print("-" * 50)
