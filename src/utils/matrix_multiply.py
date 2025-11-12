def _dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def _matmul_2d(m1, m2):
    m1_rows, m1_cols = len(m1), len(m1[0])
    m2_rows, m2_cols = len(m2), len(m2[0])

    if m1_cols != m2_rows:
        raise ValueError("Incompatible matrix dimensions for multiplication.")

    m2_T = [list(row) for row in zip(*m2)]
    
    result = [[0] * m2_cols for _ in range(m1_rows)]
    for i in range(m1_rows):
        for j in range(m2_cols):
            result[i][j] = _dot_product(m1[i], m2_T[j])
    return result

def _matmul_3d(batch1, batch2):
    return [_matmul_2d(m1, m2) for m1, m2 in zip(batch1, batch2)]

def matmul(matrix1, matrix2):
    is_3d_m1 = len(matrix1) > 0 and isinstance(matrix1[0], list) and isinstance(matrix1[0][0], list)
    is_3d_m2 = len(matrix2) > 0 and isinstance(matrix2[0], list) and isinstance(matrix2[0][0], list)

    batch1 = matrix1 if is_3d_m1 else [matrix1]
    batch2 = matrix2 if is_3d_m2 else [matrix2]

    if len(batch1) > 1 and len(batch2) == 1:
        batch2 = [batch2[0]] * len(batch1)

    if len(batch1) != len(batch2):
        raise ValueError("Batch sizes must be equal or 1 for broadcasting.")

    result_3d = _matmul_3d(batch1, batch2)

    return result_3d if is_3d_m1 else result_3d[0]
