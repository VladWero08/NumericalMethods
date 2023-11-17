import numpy as np
import matplotlib.pyplot as plt

def rgb_to_gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def get_k_rank_image(rank: int, U, S, V):
    U = U[:, :rank] 
    S = np.diag(S[:rank])
    V = V[:rank, :]
    
    return U @ S @ V

def get_lower_rank_images(image):
    ranks = [10, 50, 100]

    U, S, V = np.linalg.svd(image)

    for rank in ranks:
        image_lower_rank = get_k_rank_image(rank, U, S, V)
        plt.imshow(image_lower_rank)
        plt.show()

if __name__ == "__main__":
    image = plt.imread("adventure_time.jpg")
    image = rgb_to_gray(image)
    get_lower_rank_images()