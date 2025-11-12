import math

def sinusoidal_positional_encoding(seq_len, embedding_dim):
    positional_encoding = []
    for pos in range(seq_len):
        encoding = []
        for i in range(embedding_dim):
            angle_rate = 1 / (10000 ** (2 * (i // 2) / embedding_dim))
            angle = pos * angle_rate
            if i % 2 == 0:
                encoding.append(math.sin(angle))
            else:
                encoding.append(math.cos(angle))
        positional_encoding.append(encoding)
    return positional_encoding
