from . import softmax

def masked_softmax(vector, mask):
    masked_vector = [v if m else float('-inf') for v, m in zip(vector, mask)]
    softmax_vector = softmax.softmax(masked_vector)
    return softmax_vector
