def add_matrices(m1, m2):
    if not isinstance(m1, list) or not isinstance(m2, list):
        return m1 + m2

    is_m1_tensor = len(m1) > 0 and isinstance(m1[0], list)
    is_m2_vector = not (len(m2) > 0 and isinstance(m2[0], list))
    
    if is_m1_tensor and (is_m2_vector or len(m1) != len(m2)):
        return [add_matrices(sub_m1, m2) for sub_m1 in m1]

    if len(m1) != len(m2):
        raise ValueError(f"Matrices must have the same dimensions to be added, but got shapes {len(m1)} and {len(m2)}.")
    
    return [add_matrices(sub_m1, sub_m2) for sub_m1, sub_m2 in zip(m1, m2)]
