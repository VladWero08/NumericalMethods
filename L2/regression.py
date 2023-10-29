import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def solve_regression(file: str) -> {}:
    '''
        Given a .csv file that contains data f(z) = w for a regression problem, 
        solve the regression with least squares of Ax=b, and return a
        dictionary for each parameter in the equation Ax=b, where:

        -> A = independent values
        -> x = (α, β) solution of least squares for `f(x) = α * x + β
        that approximates f(z)=w
        -> b = dependent values
    '''

    try:
        # matrix with 2 columns, 
        # 1st column: `z` values
        # 2nd column: `w` values
        regression_data = np.genfromtxt(file, delimiter=",")
    except:
        print(f"Problem when reading {file}!")
        return []
    
    # we will solve the regression problem with
    # least squares: Ax = b, where
    # A is "training" data and b is the "testing" data
    A = regression_data[:, 0]
    A_extra_column = [1] * A.size
    A = np.column_stack((A, A_extra_column))
    b = regression_data[:, 1]

    # solve the system
    x, _, _, _ = lstsq(A, b)

    return {
        "A": A,
        "x": x,
        "b": b
    }

def generate_graphic_for_regression(regression_data: {}) -> None:
    '''
        Generate a graphic with matplotlib that contains 
        the points on which the regression solution is based 
        and also the regression line.
    '''
    # f(z) = w values
    z = regression_data["A"][:, 0]
    w = regression_data["b"]

    # extract alfa and beta from the regression solution
    alfa, beta = regression_data["x"][0], regression_data["x"][1]
    # depending on the input, calculate the function value
    # with alfa and beta
    regression = [alfa * z_element + beta for _, z_element in enumerate(z)]

    # plot desired points
    for index in range(z.size):
        plt.plot(z[index], w[index], marker="x", markersize=2, markerfacecolor="green")

    # plot generated regression line
    plt.plot(z, regression)

    # add labels and a title
    plt.xlabel('z')
    plt.ylabel('w')
    plt.title('Simple regression')

    # show the plot
    plt.show()

regression_solution = solve_regression("regresie.csv")
generate_graphic_for_regression(regression_solution)