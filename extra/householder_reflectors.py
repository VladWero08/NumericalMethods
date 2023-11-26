import numpy as np

def eliminate_2D(x: list):
    """
    Given a 2D vector, eliminate the second
    axis values using a Householder reflector U:
    
    `x` = [x1 x2]^T ---> `Ux` = [x1 0]^T,
    where ^T means the transpose of the vector
    """
    U = np.array([[1, 0], [-1 * x[1] / x[0], 1]])
    x = np.array(x)
    y = np.dot(U, x)

    return y


def eliminate_ND(x: list):
    """
    Given a ND vector, elimintate [2, N] axis
    values using a Householder reflector U:

    `x` = [x1 x2 .. xN]^T ---> `Ux` = [x1 0 ... 0]^T,
    where ^T means the transpose of the vector    
    """
    # get the dimension of x
    N = len(x)
    # start with U as the identity matrix
    U = np.identity(N)
    x = np.array(x)

    for i in range(1, N):
        # for every row from 2 -> N
        for j in range(i):
            # for every column from 1 -> i - 1,
            # calculate the value of the reflector 
            # on position [i][j] by substracting the 
            # contribution of the previous elements (1 -> i - 1) 
            U[i][j] = (-1 / i) * x[i] / x[j]

    y = np.dot(U, x)
    return y