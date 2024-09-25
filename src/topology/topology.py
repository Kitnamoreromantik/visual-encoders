from typing import Tuple

import numpy as np
import torch
from torch.linalg import svd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def calculate_anisotropy(embeddings: torch.Tensor) -> float:
    """
    Anisotropy of embedding vectors.

    Args:
        embeddings (torch.Tensor): (n_samples x n_features) tensor of embeddings.

    Returns:
        anisotropy (float):
            Anisotropy as the ratio of the largest eigenvalue to the sum of all eigenvalues.
    """
    
    centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, singular_values, _ = svd(centered_embeddings, full_matrices=False)
    covariance_eigenvalues = singular_values.square() / (embeddings.shape[0] - 1)
    anisotropy = covariance_eigenvalues.max().item() / covariance_eigenvalues.sum().item()
    
    return anisotropy


def calculate_intrinsic_dimension(embeddings: torch.Tensor, 
                                  fraction: float = 0.9, 
                                  verbose: bool = False) -> tuple:
    """
    Estimate the intrinsic dimension of a system embeddings.

    Args:
        embeddings (torch.Tensor): 2D array of shape (n, d) where n is the number of points.
        fraction (float): Fraction of the data considered for the estimation (default: 0.9).
        verbose (bool): If True, prints additional information.

    Returns:
        tuple: Contains:
            - x (np.ndarray): log(mu)
            - y (np.ndarray): -(1 - F(mu))
            - reg_coef (float): Slope of the regression line (intrinsic dimension estimate).
            - r (float): Determination coefficient of y ~ x.
            - p_val (float): p-value of y ~ x.
    """
    
    # Compute distance matrix and sort
    distance_matrix = squareform(pdist(embeddings, 'euclidean'))
    sorted_distances = np.sort(distance_matrix, axis=1)

    # Extract k1 and k2
    k1, k2 = sorted_distances[:, 1], sorted_distances[:, 2]

    # Identify problematic points
    zeros = np.where(k1 == 0)[0]
    degeneracies = np.where(k1 == k2)[0]
    good_indices = np.setdiff1d(np.arange(sorted_distances.shape[0]),
                                np.union1d(zeros, degeneracies))

    if verbose:
        print(f'Found {zeros.shape[0]} elements with r1 = 0: {zeros}')
        print(f'Found {degeneracies.shape[0]} elements with r1 = r2: {degeneracies}')
        print(f'Fraction of good points: {good_indices.shape[0] / sorted_distances.shape[0]}')

    # Filter k1 and k2
    k1, k2 = k1[good_indices], k2[good_indices]
    
    # Number of points for regression
    n_points = int(np.floor(good_indices.shape[0] * fraction))

    # Calculate mu and empirical distribution Femp
    N = good_indices.shape[0]
    mu = np.sort(k2 / k1)
    Femp = np.arange(1, N+1, dtype=np.float64) / N

    # Take logs (excluding the last element to avoid log(0))
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # Linear regression
    regr = LinearRegression(fit_intercept=False)
    regr.fit(x[:n_points, np.newaxis], y[:n_points, np.newaxis])
    r, p_val = pearsonr(x[:n_points], y[:n_points])

    return x, y, regr.coef_[0][0], r, p_val


def get_topology_characteristics(embeddings: torch.Tensor, 
                                 num_resamplings: int, 
                                 num_data_points: int) -> Tuple[float, float, float, float]:
    """
    Calculate intrinsic dimension and anisotropy estimates for a set of embeddings.

    Args:
        embeddings (torch.Tensor):
            Tensor of shape (n_samples, n_features) representing the embeddings.
        num_resamplings (int):
            Number of times to resample the data.
        num_data_points (int):
            Number of data points to use in each resampling.

    Returns:
        Tuple[float, float, float, float]:
            ID: 
                Intrinsic dimension estimate.
            ID_error: 
                Standard deviation of intrinsic dimension estimates.
            anisotropy: 
                Mean anisotropy value.
            anisotropy_error: 
                Standard deviation of anisotropy estimates.
    """

    ID_estimates = np.zeros(num_resamplings)  # Initialize array for ID
    anisotropy_estimates = np.zeros(num_resamplings)  # Initialize array for anisotropy estimates

    for i in range(num_resamplings):
        perm_idx = np.random.permutation(embeddings.shape[0])[:num_data_points]
        sampled_embeddings = embeddings[perm_idx, :]

        _, _, regr_slope, r, _ = calculate_intrinsic_dimension(sampled_embeddings)
        ID_estimates[i] = regr_slope
        anisotropy_estimates[i] = calculate_anisotropy(sampled_embeddings)

    # Calculate means and standard deviations
    ID = ID_estimates.mean()
    ID_error = ID_estimates.std()

    anisotropy = anisotropy_estimates.mean()
    anisotropy_error = anisotropy_estimates.std()

    return ID, ID_error, anisotropy, anisotropy_error
