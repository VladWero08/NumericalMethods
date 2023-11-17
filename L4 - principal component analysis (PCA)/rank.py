import numpy as np

def matrix_noise_analyzation(m: int, r: int):
    """
    Given dimensions for a matrix `A m*r`, generate
    a random matrix, append to it some columns of its initial
    form (to generate a matrix with dependent columns), add noise
    to the matrix with dependent columns.
    
    For this 3 matrixs analyze the rank and the singular values 
    `(from SVD)` of the matrix with noise.
    """
    matrix = np.random.randn(m, r)
    rank_matrix = np.linalg.matrix_rank(matrix)

    column = matrix[:, 1]
    # concatenate some dependent columns to
    # the matrix in order to have a matrix 
    # that does not have a maximal rank
    matrix = np.column_stack((matrix, column * 2))
    matrix = np.column_stack((matrix, column * 4))
    matrix = np.column_stack((matrix, column * 6))
    matrix = np.column_stack((matrix, column * 8))
    rank_matrix_new = np.linalg.matrix_rank(matrix)

    # add Gaussian noise to the matrix with
    # dependent columns
    noise = np.random.normal(0, 0.2, matrix.shape)
    matrix = np.add(matrix, noise)
    rank_matrix_with_noise = np.linalg.matrix_rank(matrix)

    print(f"Matrix rank: {rank_matrix}")
    print(f"Matrix rank with dependent columns: {rank_matrix_new}")
    print(f"Matrix rank with dependent columns and small Gaussian noise: {rank_matrix_with_noise}")

    U, S, V = np.linalg.svd(matrix)
    print(f"Singular values of matrix with noise: {S}")
    print(f"Relevant singular values: {S[S > 1]}")

if __name__ == "__main__":
    matrix_noise_analyzation(10, 4)