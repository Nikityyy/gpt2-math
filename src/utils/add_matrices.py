def add_matrices(matrix1, matrix2):
    # check if dimensions match
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions to be added.")
    
    result = []
    for i in range(len(matrix1)):
        result_row = []
        for j in range(len(matrix1[0])):
            result_row.append(matrix1[i][j] + matrix2[i][j])
        result.append(result_row)
    
    return result
