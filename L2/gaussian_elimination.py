import numpy as np

def get_upper_triangular(n, matrix_coeff, array_sol):
    """Given a dimension, matrix of coefficients, array of solution values,
    transform the matrix of coefficients into its upper triangular form, respectively
    the array of solution values into its corresponding form."""

    for column in range(n):
        # for each line, create the matrix which
        # will transform values from the first column into 0s
        M_identity = np.identity(n)

        for line in range(column + 1, n):
            # compute multiplicators matrix M_column
            M_identity[line][column] = -1 * matrix_coeff[line][column] / matrix_coeff[column][column]

        matrix_coeff = np.matmul(M_identity, matrix_coeff)
        array_sol = np.matmul(M_identity, array_sol)

    return matrix_coeff, array_sol

def utris(n, matrix_coeff, array_sol):
    """Given a dimension, matrix of coefficients in triangular form, 
    array of solution values, solve the system related to the parameters."""

    x = array_sol

    for line in range(n - 1, -1, -1):
        # being upper triangular matrix, start computing
        # the solution from the bottom right corner
        for column in range(n - 1, line, -1):
            x[line] = x[line] - matrix_coeff[line][column] * x[column]

        x[line] = x[line] / matrix_coeff[line][line]

    return x

def generate_and_solve_system(dimension):
    """Generate an equation Ax = B and solve it using
    Gaussian Elimination."""

    A = np.random.randn(dimension, dimension)
    B = np.random.randn(dimension)
    numpy_solution = np.linalg.solve(A, B)   

    A, B = get_upper_triangular(dimension, A, B)
    solution = utris(dimension, A, B)

    print(f"numpy: {numpy_solution}")
    print(f"gauss: {solution}")

generate_and_solve_system(6)