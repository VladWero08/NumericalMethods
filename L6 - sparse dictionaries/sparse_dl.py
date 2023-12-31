import numpy as np
import matplotlib.pyplot as plt

from dictlearn import DictionaryLearning
from dictlearn import methods
from matplotlib import image
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.preprocessing import normalize

# CONSTANTS
# #########
patch_size = (8, 8)
no_patches = 796
# level of sparsity for the dictionary learning
sparse = 6
# number of atoms in a dictionary
no_atoms = 256
# number of iterations for the dictionary learning
no_dl_iterations = 50
# standard deviation to add noise for image
sigma = 5
# path to the image that will be denoised
image_path = "./regular_show.jpg"

def rgb_to_gray(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def get_patches(image: np.ndarray, patch_size: (int, int)) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an grayscaled `image` and a `patch size` of
    `(N x N)`, create patches from that, and return a list of patches.
    """
    # get the patches for training
    Y_patches = extract_patches_2d(image, patch_size)
    # vectorize them, to have N patches of size patch * patch
    Y_patches = Y_patches.reshape(Y_patches.shape[0], -1)
    # subtitue the mean from each patch
    Y_patches_means = np.transpose(Y_patches).mean(axis=0)
    Y_patches = Y_patches - Y_patches_means[:, np.newaxis]

    return (Y_patches, Y_patches_means)

def train_dictionary(
    train_patches: np.ndarray, 
    transform_patches: np.ndarray, 
    transform_patches_means: np.ndarray,
) -> np.ndarray:
    """
    Given a set of `train patches`, and the `patches to transform`,
    using K-SVD to rebuild the `transform patches` into their original form.

    Return the transformed patches.
    """

    # generate a random dictionary to start with
    D0 = np.random.randn(train_patches.shape[0], train_patches.shape[1])
    # normalize the dictionary
    D0 = normalize(D0, axis=0, norm="max")

    # use DL to train the dictionary to fit a random
    # sample from the whole array of patches
    dictionary_learning = DictionaryLearning(
        max_iter=no_dl_iterations,
        fit_algorithm="ksvd",
        n_nonzero_coefs=sparse,
        code_init=None,
        dict_init=D0,
        params=None,
        data_sklearn_compat=False
    )
    print("Dictionary is fitting the data...")
    dictionary_learning.fit(train_patches)
    
    # used the trained dictionary to transform all
    # patches back into their initial value
    print("Dictionary is transforming the data...")

    Xc, _ = methods.omp(train_patches[:no_patches], dictionary_learning.D_, n_nonzero_coefs=sparse)
    Yc = np.dot(dictionary_learning.D_, Xc)

    for batch in range(no_patches, transform_patches.shape[0], no_patches):
        Xc_batch, _ = methods.omp(transform_patches[batch:batch + no_patches], dictionary_learning.D_, n_nonzero_coefs=sparse)
        Yc_batch = np.dot(dictionary_learning.D_, Xc_batch)
        Yc = np.concatenate((Yc, Yc_batch), axis=0)

    Yc = Yc + transform_patches_means[:, np.newaxis]
    print("Patches have been reconstructed!")

    return Yc

def psnr(img1: np.ndarray, img2: np.ndarray):
    """Peak Signal to Noise Ratio between two images."""

    mse = np.mean((img1 - img2) ** 2)

    if(mse == 0):
        return 0
    
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def display(original_img: np.ndarray, noisy_img: np.ndarray, cleared_img: np.ndarray) -> None:
    """Display all images provided as parameters, where to the
    original image has been added noise, and the cleread image had the
    noise removed using Sparse Dictionary Learning."""

    plt.imshow(original_img)
    plt.show()

    plt.imshow(noisy_img)
    plt.show()

    plt.imshow(cleared_img)
    plt.show()

# load the RGB image and transform it to grayscale
I = image.imread(image_path)
I = rgb_to_gray(I)

# add noise to the image
(m1, m2) = I.shape
noise = sigma * np.random.randn(m1, m2)
Inoisy = I + noise

# generate and extract a random number of patches
# from the image that has noise
Y_patches, Y_patches_means = get_patches(Inoisy, patch_size)
Y = np.random.choice(Y_patches.shape[1], no_patches)
Y = Y_patches[Y, :]

Yc = train_dictionary(
    train_patches=Y, 
    transform_patches=Y_patches,
    transform_patches_means=Y_patches_means
)
Yc = Yc.reshape(Yc.shape[0], patch_size[0], patch_size[0])
Ic = reconstruct_from_patches_2d(Yc, Inoisy.shape)

display(I, Inoisy, Ic)
print(f"PSNR between original and noisy: {psnr(I, Inoisy)}")
print(f"PSNR between original and cleared: {psnr(I, Ic)}")