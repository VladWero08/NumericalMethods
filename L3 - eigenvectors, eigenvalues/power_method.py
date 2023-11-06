import numpy as np
import matplotlib.pyplot as plt

def get_most_significant_eigenvector(A: np.array) -> np.array:
    # get the eigenvalues and eigenvectors of the matrix
    eigen = np.linalg.eigh(A)
    eigenvalues, eigenvectors = eigen[0], eigen[1]

    # transform the eigenvalues vector into a vector
    # with absolute values
    eigenvalues = np.abs(eigenvalues)

    # find the index for the biggest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)

    # return the most significant eigenvector
    return eigenvectors[:, max_eigenvalue_index]

def power_metod(
    A: np.array, 
    y: np.array, 
    max_interations: int = 10000,
    tolerance: int = 0.00001
) -> np.array: 
    iter = 0
    error = 1

    # A * y will be computed until it reaches the maximum number
    # of iterations OR until it the transformation A * y will not
    # significantly modified the current y vector
    while iter < max_interations and error > tolerance:
        # normalize the target vector
        y = y / np.linalg.norm(y)
        # compute A * y again
        z = np.matmul(A, y)
        # normalize the newly computed vector 
        z = z / np.linalg.norm(z)

        # compute the dot product betwenn the computer
        # vector and the previous one 
        zt_dot_y = np.dot(z, y)

        # it will indicate if the vector `z` was
        # significantly changed
        error = abs(1 - abs(zt_dot_y))

        y = z
        iter += 1

    return y
    
if __name__ == "__main__":
    dimension = 6
    A = np.random.rand(dimension, dimension)
    y = np.random.rand(dimension)

    print(get_most_significant_eigenvector(A))
    print(power_metod(A, y))