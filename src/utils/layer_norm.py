def init_layer_norm(normalized_shape_dim):
    gamma = [1.0] * normalized_shape_dim  # weight
    beta = [0.0] * normalized_shape_dim   # bias
    return gamma, beta

def _layer_norm_1d(x, gamma, beta, epsilon=1e-5):
    if not x: return []
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    denom = (variance + epsilon) ** 0.5
    return [((xi - mean) / denom) * g + b for xi, g, b in zip(x, gamma, beta)]

def layer_norm(tensor, gamma, beta, epsilon=1e-5):
    if tensor and isinstance(tensor[0], list):
        return [layer_norm(sub_tensor, gamma, beta, epsilon) for sub_tensor in tensor]
    else:
        return _layer_norm_1d(tensor, gamma, beta, epsilon)
