import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def normal_distribution(mean: int, dispersion: int):
    """
    Given a mean and a dispersion, plots a graph
    with the `normal distribution` given by these parameters.
    """
    x = np.arange(mean - 50, mean + 50, 0.5)
    distribution = stats.norm.pdf(x, mean, dispersion)

    plt.plot(x, distribution)
    plt.show()

def verosimility(x: int, mean: int, dispersion: int):
    """
    Returns the verosimility of `x` to be part of the 
    `normal distribution` given by the `mean` and `dispersion`.
    """
    frac = 1 / np.sqrt(2 * np.pi * (dispersion ** 2))
    exponent = -1 * ((x - mean) ** 2) / (2 * (dispersion ** 2))

    return frac * (np.e ** exponent)

def verosimility_array(x: np.ndarray, mean: int, dispersion: int):
    """
    Returns the verosimility product between each element in the array
    to be part of the `normal distribution` given by the `mean` and `dispersion`.
    """
    verosimilities = [verosimility(element, mean, dispersion) for element in x]
    probability = 1

    for verosimility_ in verosimilities:
        probability *= verosimility_

    return probability

def check_verosimility(x: int, mean: int, dispersion):
    """
    Calculates the verosimility of `x` to be part of
    the `normal distribution` given by the `mean` and `dispersion`
    and checks its validity using `scipy.stats.norm.pdf()`.
    """
    verosimility_ = verosimility(x, mean, dispersion)

    print(f"Got verosimility {verosimility_} for value {x} in normal distribution of mean {mean} and dispersion {dispersion}.")

    if verosimility_ == stats.norm.pdf(x, mean, dispersion):
        print("Which is valid...")
    else:
        print("Which is invalid...")

def probability_a_priori(
    mean: int, 
    dispersion: int, 
    mean_distribution = (100, 50),
    dispersion_interval = (1, 70),
):
    """
    Computes the probability of the `mean` to be in
    the normal distribution given by `mean_distribution` and
    the `dispersion` to be in the uniform distribution `dispersion_interval`
    given by the `dispersion_interval`.
    """

    mean_probability = stats.norm.pdf(mean, mean_distribution[0], mean_distribution[1])
    dispersion_probability = stats.uniform.pdf(dispersion, dispersion_interval[0], dispersion_interval[1])
    
    return mean_probability * dispersion_probability

def probability_a_posteriori(data: np.ndarray, mean: int, dispersion: int):
    """
        Computes the a posteriori probability of the `mean` and `dispersion`
        to be the correct ones for the "parent" dataset of `data`.

        ### Parameters
        - `data`: sample of data from a bigger data set
        - `mean`: mean to be tested for data 
        - `dispersion`: dispersion to be tested for data
    """
    return verosimility_array(data, mean, dispersion) * probability_a_priori(mean, dispersion)

def optimal(
    data: np.ndarray,
    means: np.ndarray,
    dispersions: np.ndarray,
):
    """
        Given an array of `means` and `dispersions` and a target `data` array, 
        computes the a posteriori probability for each combination of mean and dispersion
        in order to find the best fit for the given data.
    """

    max_ = {
        "probability": -1,
        "mean": 0,
        "dispersion": 0,
    }

    for mean in means:
        for dispersion in dispersions:
            probability_mean = probability_a_posteriori(
                data=data,
                mean=mean,
                dispersion=dispersion,
            )

            print(f"Probability mean = {probability_mean} for mean {mean} and dispersion {dispersion}")

            if probability_mean > max_["probability"]:
                max_["probability"] = probability_mean
                max_["dispersion"] = dispersion
                max_["mean"] = mean

    print()
    print(f"Optimal choice: mean {max_['mean']}, dispersion {max_['dispersion']}")

if __name__ == "__main__":
    house_prices = [82, 106, 120, 68,83, 89, 130, 92, 99, 89]
    mean = 90
    dispersion = 10

    means = [70, 75, 80, 85, 90, 95, 100]
    dispersions = [5, 10, 15, 20]

    optimal(
        data=house_prices,
        means=means,
        dispersions=dispersions
    )