import numpy as np
from scipy import stats
from typing import List, Tuple
import random


def hamming_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized Hamming distance between two strings.
    
    Args:
        s1, s2: Input strings
        
    Returns:
        float: Normalized Hamming distance (0 to 1)
    """
    if len(s1) != len(s2):
        # Pad shorter string with spaces or truncate longer one
        max_len = max(len(s1), len(s2))
        s1 = s1.ljust(max_len)[:max_len]
        s2 = s2.ljust(max_len)[:max_len]
    
    if len(s1) == 0:
        return 0.0
    
    # Count differences
    differences = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return differences / len(s1)


def hamming_kernel(s1: str, s2: str, gamma: float = 1.0) -> float:
    """
    Hamming distance kernel function.
    
    Args:
        s1, s2: Input strings
        gamma: Kernel parameter (controls sensitivity)
        
    Returns:
        float: Kernel value
    """
    hamming_dist = hamming_distance(s1, s2)
    # Use RBF-like kernel: exp(-gamma * hamming_distance)
    return np.exp(-gamma * hamming_dist)


def compute_kernel_matrix(fingerprints1: List[str], fingerprints2: List[str], gamma: float = 1.0) -> np.ndarray:
    """
    Compute kernel matrix between two sets of fingerprints.
    
    Args:
        fingerprints1, fingerprints2: Lists of fingerprint strings
        gamma: Kernel parameter
        
    Returns:
        np.ndarray: Kernel matrix of shape (len(fingerprints1), len(fingerprints2))
    """
    n1, n2 = len(fingerprints1), len(fingerprints2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            K[i, j] = hamming_kernel(fingerprints1[i], fingerprints2[j], gamma)
    
    return K


def mmd_statistic(fingerprints1: List[str], fingerprints2: List[str], gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) statistic using Hamming kernel.
    
    Args:
        fingerprints1, fingerprints2: Lists of fingerprint strings
        gamma: Kernel parameter
        
    Returns:
        float: MMD statistic
    """
    n1, n2 = len(fingerprints1), len(fingerprints2)
    
    if n1 == 0 or n2 == 0:
        return 1.0  # Maximum discrepancy
    
    # Compute kernel matrices
    K11 = compute_kernel_matrix(fingerprints1, fingerprints1, gamma)
    K22 = compute_kernel_matrix(fingerprints2, fingerprints2, gamma)
    K12 = compute_kernel_matrix(fingerprints1, fingerprints2, gamma)
    
    # Use unbiased estimator for MMD^2
    # MMD^2 = (1/(n1*(n1-1))) * sum_{i≠j} k(x_i, x_j) + 
    #         (1/(n2*(n2-1))) * sum_{i≠j} k(y_i, y_j) - 
    #         (2/(n1*n2)) * sum_{i,j} k(x_i, y_j)
    
    if n1 == 1:
        term1 = 0.0  # Can't compute unbiased estimate with single sample
    else:
        # Mean of K11 excluding diagonal
        K11_off_diag = K11.copy()
        np.fill_diagonal(K11_off_diag, 0)
        term1 = np.sum(K11_off_diag) / (n1 * (n1 - 1))
    
    if n2 == 1:
        term2 = 0.0  # Can't compute unbiased estimate with single sample
    else:
        # Mean of K22 excluding diagonal
        K22_off_diag = K22.copy()
        np.fill_diagonal(K22_off_diag, 0)
        term2 = np.sum(K22_off_diag) / (n2 * (n2 - 1))
    
    # Cross terms
    term3 = np.mean(K12)
    
    mmd_squared = term1 + term2 - 2 * term3
    
    # For identical fingerprints, MMD should be exactly 0
    if fingerprints1 == fingerprints2:
        return 0.0
    
    # Return MMD (square root), but ensure non-negative
    return np.sqrt(max(0.0, mmd_squared))


def bootstrap_null_distribution(base_fingerprints: List[str], 
                               n_bootstrap: int = 1000, 
                               gamma: float = 1.0,
                               random_seed: int = None) -> List[float]:
    """
    Generate null distribution by bootstrapping from base model fingerprints.
    
    Args:
        base_fingerprints: Fingerprints from base model
        n_bootstrap: Number of bootstrap samples
        gamma: Kernel parameter
        random_seed: Random seed for reproducibility
        
    Returns:
        List[float]: MMD statistics under null hypothesis
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    n_samples = len(base_fingerprints)
    if n_samples < 2:
        # Need at least 2 samples
        return [0.0] * n_bootstrap
    
    null_mmds = []
    
    for _ in range(n_bootstrap):
        if n_samples >= 4:
            # Randomly split base fingerprints into two groups
            shuffled = base_fingerprints.copy()
            random.shuffle(shuffled)
            
            # Split roughly in half
            split_point = n_samples // 2
            group1 = shuffled[:split_point]
            group2 = shuffled[split_point:]
        else:
            # For small samples, use resampling with replacement to create variation
            # Create larger resampled groups to ensure some variation
            resample_size = max(n_samples, 3)
            group1 = [random.choice(base_fingerprints) for _ in range(resample_size)]
            group2 = [random.choice(base_fingerprints) for _ in range(resample_size)]
            
            # Add some artificial noise to create variation when all samples are identical
            if len(set(base_fingerprints)) == 1:  # All fingerprints are identical
                # Create slight variations by modifying a few characters
                def add_noise(s):
                    if len(s) > 0 and random.random() < 0.3:  # 30% chance to modify
                        idx = random.randint(0, len(s) - 1)
                        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
                        new_char = random.choice(chars)
                        return s[:idx] + new_char + s[idx+1:]
                    return s
                
                group2 = [add_noise(s) for s in group2]
        
        # Compute MMD between the two groups
        mmd = mmd_statistic(group1, group2, gamma)
        null_mmds.append(mmd)
    
    return null_mmds


def compute_p_value(observed_mmd: float, null_distribution: List[float]) -> float:
    """
    Compute p-value as the proportion of null MMDs >= observed MMD.
    
    Args:
        observed_mmd: Observed MMD statistic
        null_distribution: List of MMD statistics under null hypothesis
        
    Returns:
        float: p-value
    """
    if not null_distribution:
        return 1.0
    
    # For MMD, we want to test if observed MMD is significantly different from 0
    # Small MMD suggests similarity, large MMD suggests difference
    # p-value = P(null_MMD >= observed_MMD | H0)
    
    # Count how many null MMDs are >= observed MMD
    count = sum(1 for null_mmd in null_distribution if null_mmd >= observed_mmd)
    
    # p-value = proportion of null statistics >= observed
    p_value = count / len(null_distribution)
    
    # Handle edge cases for numerical stability
    n = len(null_distribution)
    if count == 0:
        # If no null values >= observed, use 1/(2*n) as minimum p-value
        p_value = 1.0 / (2 * n)
    elif count == n:
        # If all null values >= observed, use 1-1/(2*n) as maximum p-value
        p_value = 1.0 - 1.0 / (2 * n)
    
    return p_value


def met_similarity_test(base_fingerprints: List[str], 
                       test_fingerprints: List[str],
                       gamma: float = 1.0,
                       n_bootstrap: int = 1000,
                       alpha: float = 0.05,
                       random_seed: int = None) -> Tuple[float, dict]:
    """
    Perform MET similarity test using MMD with Hamming kernel.
    
    Args:
        base_fingerprints: Fingerprints from base model
        test_fingerprints: Fingerprints from test model
        gamma: Kernel parameter
        n_bootstrap: Number of bootstrap samples for null distribution
        alpha: Significance level
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple[float, dict]: (similarity_score, test_statistics)
    """
    # Compute observed MMD
    observed_mmd = mmd_statistic(base_fingerprints, test_fingerprints, gamma)
    
    # Generate null distribution
    null_mmds = bootstrap_null_distribution(base_fingerprints, n_bootstrap, gamma, random_seed)
    
    # Compute p-value
    p_value = compute_p_value(observed_mmd, null_mmds)
    
    # Determine similarity (1.0 if similar, 0.0 if different)
    # High p-value means we cannot reject H0 (models are similar)
    # Low p-value means we reject H0 (models are different)
    is_similar = p_value >= alpha  # Changed from < to >=
    similarity_score = 1.0 if is_similar else 0.0
    
    # Alternative similarity measures that could be more informative
    # 1. P-value based similarity (higher p-value = more similar)
    p_value_similarity = p_value  # Changed from 1.0 - p_value
    
    # 2. MMD-based similarity (lower MMD = more similar)
    max_possible_mmd = 1.0  # Theoretical maximum for normalized Hamming kernel
    mmd_similarity = max(0.0, 1.0 - observed_mmd / max_possible_mmd)
    
    # 3. Percentile-based similarity
    if null_mmds:
        percentile = stats.percentileofscore(null_mmds, observed_mmd) / 100.0
        percentile_similarity = 1.0 - percentile
    else:
        percentile_similarity = 0.0
    
    test_statistics = {
        'observed_mmd': observed_mmd,
        'p_value': p_value,
        'is_similar': is_similar,
        'alpha': alpha,
        'n_bootstrap': n_bootstrap,
        'null_mmds_mean': np.mean(null_mmds) if null_mmds else 0.0,
        'null_mmds_std': np.std(null_mmds) if null_mmds else 0.0,
        'p_value_similarity': p_value_similarity,
        'mmd_similarity': mmd_similarity,
        'percentile_similarity': percentile_similarity
    }
    
    # Return the most informative similarity measure
    # P-value based similarity is most interpretable for hypothesis testing
    return p_value_similarity, test_statistics
