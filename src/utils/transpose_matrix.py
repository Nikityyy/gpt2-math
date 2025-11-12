def _transpose_2d(matrix):
    return [list(row) for row in zip(*matrix)]

def _transpose_3d(batch):
    return [_transpose_2d(matrix) for matrix in batch]

def transpose_matrix(matrix):
    if not matrix:
        return []

    is_3d = isinstance(matrix[0], list) and isinstance(matrix[0][0], list)
    
    if is_3d:
        return _transpose_3d(matrix)
    else:
        return _transpose_2d(matrix)
