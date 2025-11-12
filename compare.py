import numpy as np
import src.utils as utils

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

if __name__ == "__main__":
    mat1 = [[1, 2, 3],
            [4, 5, 6]]
    
    mat2 = [[7, 8],
            [9, 10],
            [11, 12]]
    
    mat3 = [[1, 2, 3],
            [4, 5, 6]]
    
    mat4 = [[7, 8, 9],
            [10, 11, 12]]
    
    compare_matmul(mat1, mat2)
    compare_add_matrices(mat3, mat4)
