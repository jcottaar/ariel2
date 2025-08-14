import numpy as np
import cupy as cp

def estimate_noise(ts, window_size=10, degree=2, combine_method='rms'):
    """
    Estimate the noise level of a 1D time-series by:
      1. Sliding a window of `window_size` over the data.
      2. Fitting a polynomial of given `degree` to each window.
      3. Computing the RMS of the residuals in each window.
      4. Correcting for the bias introduced by fitting.
      5. Combining all corrected RMS values via:
         - 'rms': sqrt(mean(rms_corrected**2))  ← unbiased for σ
         - 'mean': mean(rms_corrected)           (biased low)
         - 'median': median(rms_corrected)       (robust but biased)
    
    Parameters
    ----------
    ts : array-like
        The time-series data.
    window_size : int, optional
        Number of consecutive points per window (default is 10).
    degree : int, optional
        Polynomial degree to fit (default is 2, for quadratic).
    combine_method : {'rms', 'mean', 'median'}, optional
        How to combine corrected window RMS values into a single noise estimate.
    
    Returns
    -------
    float
        The combined, bias-corrected noise estimate.
    """
    ts = np.asarray(ts)
    N = ts.size
    if N < window_size:
        raise ValueError(f"Time series length ({N}) is shorter than window size ({window_size}).")
    
    # sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(ts, window_size)
    
    # precompute Vandermonde and its pseudoinverse
    x = np.arange(window_size)
    V = np.vander(x, degree + 1)
    pinv = np.linalg.pinv(V)
    
    # fit and compute residual RMS per window
    coeffs = windows.dot(pinv.T)
    fitted = coeffs.dot(V.T)
    residuals = windows - fitted
    rms = np.sqrt(np.mean(residuals**2, axis=1))
    
    # bias correction for each window
    p = degree + 1
    correction = np.sqrt(window_size / (window_size - p))
    rms_corr = rms * correction
    
    # combine
    if combine_method == 'rms':
        return float(np.sqrt(np.mean(rms_corr**2)))
    elif combine_method == 'mean':
        return float(np.mean(rms_corr))
    elif combine_method == 'median':
        return float(np.median(rms_corr))
    else:
        raise ValueError("combine_method must be 'rms', 'mean' or 'median'")

def estimate_noise_cp(ts, window_size=10, degree=2, combine_method='rms'):
    """
    CuPy version: same behavior, runs on GPU.
    """
    ts = cp.asarray(ts)
    N = ts.size
    if N < window_size:
        raise ValueError(f"Time series length ({int(N)}) is shorter than window size ({window_size}).")

    # sliding windows on GPU
    windows = cp.lib.stride_tricks.sliding_window_view(ts, window_size)  # shape: (N - w + 1, w)

    # Vandermonde and pseudoinverse on GPU
    x = cp.arange(window_size)
    V = cp.vander(x, degree + 1)                     # (w, p)
    pinv = cp.linalg.pinv(V)                          # (p, w)

    # fit and residual RMS per window (all on GPU)
    coeffs = windows.dot(pinv.T)                      # (num_windows, p)
    fitted = coeffs.dot(V.T)                          # (num_windows, w)
    residuals = windows - fitted
    rms = cp.sqrt(cp.mean(residuals**2, axis=1))      # (num_windows,)

    # bias correction
    p = degree + 1
    correction = cp.sqrt(window_size / (window_size - p))
    rms_corr = rms * correction

    # combine and return as Python float
    if combine_method == 'rms':
        result = cp.sqrt(cp.mean(rms_corr**2))
    elif combine_method == 'mean':
        result = cp.mean(rms_corr)
    elif combine_method == 'median':
        result = cp.median(rms_corr)
    else:
        raise ValueError("combine_method must be 'rms', 'mean' or 'median'")

    return result

import cupy as cp

def estimate_noise_cov_cp(ts, window_size=10, degree=2):
    """
    Estimate an NxN covariance matrix of 'noise' across columns of a 2D time series
    by locally detrending each column with a polynomial of given degree over
    sliding windows, then computing the covariance of the residuals.

    Parameters
    ----------
    ts : (T, N) cupy.ndarray
        2D time series; N is the number of columns (variables).
    window_size : int
        Size of the sliding window along the time axis.
    degree : int
        Polynomial degree for local detrending.

    Returns
    -------
    cov : (N, N) cupy.ndarray
        Covariance of residuals across columns.
    """
    ts = cp.asarray(ts)
    if ts.ndim != 2:
        raise ValueError("ts must be 2D with shape (T, N).")
    T, N = ts.shape
    w = int(window_size)
    if T < w:
        raise ValueError(f"Time series length ({T}) is shorter than window size ({w}).")

    p = degree + 1
    if w <= p:
        raise ValueError(f"window_size ({w}) must be > degree+1 ({p}).")

    # Sliding windows over time; result shape: (T-w+1, N, w)  [window axis is LAST]
    windows = cp.lib.stride_tricks.sliding_window_view(ts, window_shape=w, axis=0)

    # Build polynomial projector onto residuals: R = I - V @ pinv(V)  (w x w)
    x = cp.arange(w)
    V = cp.vander(x, p)                 # (w, p)
    pinv = cp.linalg.pinv(V)            # (p, w)
    R = cp.eye(w, dtype=ts.dtype) - V @ pinv  # (w, w)

    # Apply residual projector along the window axis (last axis)
    # einsum does y * R with y along last dim: out[..., u] = sum_w windows[..., w] * R[w, u]
    residuals = cp.einsum('tnw,wu->tnu', windows, R)  # (T-w+1, N, w)

    # Bias correction for variance loss due to fitting p parameters
    corr = cp.sqrt(w / (w - p))
    residuals *= corr

    # Flatten window/time axes and compute covariance across columns
    Rflat = residuals.transpose(0, 2, 1).reshape(-1, N)  # ( (T-w+1)*w, N )
    Rflat -= Rflat.mean(axis=0, keepdims=True)

    L = Rflat.shape[0]
    cov = (Rflat.T @ Rflat) / (L - 1)
    return cov

import cupyx.scipy.sparse.linalg as cusparse

def nan_pca(mat, n_components, max_iter=50, tol=1e-6,
            verbose=False, c_batch=256, w_batch=256):
    """
    Perform PCA on data with missing values using alternating least squares,
    fitting directly on observed entries in chunked GPU solves to limit memory use.

    Args:
        mat (cp.ndarray): Input data, shape (n_samples, ...). May contain NaNs.
        n_components (int): Number of principal components to retain.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance on reconstruction error change.
        verbose (bool): If True, prints iteration diagnostics.
        c_batch (int): Number of features to solve per chunk in C-update.
        w_batch (int): Number of samples to solve per chunk in W-update.

    Returns:
        W (cp.ndarray): Projected scores (n_samples, n_components).
        C (cp.ndarray): Principal components (n_components, n_features).
        S_post (cp.ndarray): Post-hoc component strengths.
    """
    # Flatten per-sample features
    X = cp.array(mat, copy=True)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, -1)
    n_features = X.shape[1]

    # Prepare mask and initial fill
    mask = ~cp.isnan(X)
    X[cp.sum(mask,1) < n_components + 1, :] = 0.0
    X[:, cp.sum(mask,0) < n_components + 1] = 0.0
    mask = ~cp.isnan(X)
    X[~mask] = 0.0
    mask = mask.astype(X.dtype)

    # Random initialization
    rng = cp.random.default_rng(seed=42)
    W = rng.standard_normal((n_samples, n_components), dtype=X.dtype)
    C = rng.standard_normal((n_components, n_features), dtype=X.dtype)

    prev_error = cp.inf
    for itrr in range(1, max_iter + 1):
        # Chunked C-update
        for j0 in range(0, n_features, c_batch):
            j1 = min(j0 + c_batch, n_features)
            mask_j = mask[:, j0:j1]   # (samples, batch)
            X_j = X[:, j0:j1]
            # build batch A and B: (batch, k, k) and (batch, k)
            A2 = cp.einsum('ip,ij,iq->jpq', W, mask_j, W)
            B2 = cp.einsum('ip,ij,ij->jp', W, mask_j, X_j)
            C[:, j0:j1] = cp.linalg.solve(A2, B2[..., None]).squeeze(-1).T
        C, _ = cp.linalg.qr(C.T)
        C = C.T

        # Chunked W-update
        for i0 in range(0, n_samples, w_batch):
            i1 = min(i0 + w_batch, n_samples)
            mask_i = mask[i0:i1, :]   # (batch, features)
            X_i = X[i0:i1, :]
            A = cp.einsum('aj,ij,cj->iac', C, mask_i, C)
            B = cp.einsum('aj,ij,ij->ia', C, mask_i, X_i)
            W[i0:i1] = cp.linalg.solve(A, B[..., None]).squeeze(-1)

        # Compute error
        err = cp.linalg.norm((X - W @ C) * mask)
        if verbose:
            print(f"Iteration {itrr}: error = {err:.6f}")
        if abs(prev_error - err) < tol:
            if verbose:
                print(f"Converged at iteration {itrr}: error = {err:.6f}")
            break
        prev_error = err

    # Reshape C back and order components
    orig_shape = mat.shape
    C = C.reshape((n_components,) + orig_shape[1:])
    S_post = cp.linalg.norm(W, axis=0)
    order = cp.argsort(S_post)[::-1]
    W = W[:, order]
    C = C[order]
    S_post = S_post[order]

    return W, C, S_post

import cupy as cp

def lstsq_nanrows_normal_eq_with_pinv_sigma(
    dat: cp.ndarray,
    A: cp.ndarray,
    sigma: cp.ndarray | None = None
):
    """
    Vectorized weighted least squares via normal equations, returning coefficients
    and the weighted pseudoinverse of A.

    Parameters
    ----------
    dat   : (N, M) CuPy array, columns are RHS; rows may be entirely NaN.
    A     : (N, P) CuPy design matrix, no NaNs.
    sigma : (N,)   Optional per-row noise *std* for weighting.

    Returns
    -------
    coeffs   : (P, M) CuPy array
        Fitted coefficients for each column in `dat`.
    A_pinv_w : (P, N_valid) CuPy array
        Weighted pseudoinverse mapping valid-row inputs to coeffs:
        coeffs = A_pinv_w @ dat[mask].
        With weights w = 1/sigma, this equals (A^T W^2 A)^{-1} A^T W^2.
    mask     : (N,) boolean CuPy array
        Row mask used (True = kept).
    """
    # Keep only rows that are valid for all columns (entire-row NaNs)
    mask = ~cp.any(cp.isnan(dat), axis=1)
    A_use = A[mask]          # (N_valid, P)
    Y_use = dat[mask]        # (N_valid, M)

    if sigma is not None:
        w = 1.0 / sigma[mask]              # (N_valid,)
    else:
        w = cp.ones(A_use.shape[0], dtype=A.dtype)

    # Row-weighted matrices
    A_w = A_use * w[:, None]               # (N_valid, P)
    Y_w = Y_use * w[:, None]               # (N_valid, M)

    # Normal equations
    AtA = A_w.T @ A_w                      # (P, P)
    AtY = A_w.T @ Y_w                      # (P, M)

    # Solve for coefficients
    coeffs = cp.linalg.solve(AtA, AtY)     # (P, M)

    # Weighted pseudoinverse: (A^T W^2 A)^{-1} A^T W^2
    # Note: A^T W^2 = (A_w^T) * w[None, :]
    A_pinv_w = cp.linalg.solve(AtA, (A_w.T * w[None, :]))  # (P, N_valid)

    return coeffs, A_pinv_w, mask


import cupy as cp
from cupyx.scipy.linalg import solve_triangular

def lstsq_nanrows_gls_with_pinv(
    dat: cp.ndarray,
    A: cp.ndarray,
    cov: cp.ndarray | None = None,
    ridge: float = 0.0
):
    """
    Generalized least squares with full noise covariance.
    Assumes entire-row NaNs in `dat` (same mask across columns).

    Parameters
    ----------
    dat  : (N, M) CuPy array (rows may be all-NaN; shared across columns)
    A    : (N, P) CuPy design matrix (no NaNs)
    cov  : (N, N) CuPy covariance across ROWS (observations). If None -> OLS.
    ridge: small nonnegative scalar to stabilize AtA (adds ridge*I to AtA)

    Returns
    -------
    coeffs     : (P, M)
    A_pinv_cov : (P, N_valid)   # satisfies coeffs = A_pinv_cov @ dat[mask]
    mask       : (N,) bool
    """
    # Shared valid-row mask
    mask = ~cp.any(cp.isnan(dat), axis=1)
    A_use = A[mask]         # (N_valid, P)
    Y_use = dat[mask]       # (N_valid, M)

    if cov is None:
        # OLS
        AtA = A_use.T @ A_use
        if ridge:
            AtA = AtA + ridge * cp.eye(AtA.shape[0], dtype=AtA.dtype)
        AtY = A_use.T @ Y_use
        coeffs = cp.linalg.solve(AtA, AtY)
        A_pinv_cov = cp.linalg.solve(AtA, A_use.T)
        return coeffs, A_pinv_cov, mask

    # GLS
    Sigma_use = cov[mask][:, mask]
    # (Optional) symmetrize for numerical hygiene
    Sigma_use = 0.5 * (Sigma_use + Sigma_use.T)

    # Cholesky returns LOWER triangular L with Sigma = L @ L.T
    L = cp.linalg.cholesky(Sigma_use)

    # Whiten: A' = L^{-1} A,  Y' = L^{-1} Y  (solve triangular systems)
    A_white = solve_triangular(L, A_use, lower=True, check_finite=False)
    Y_white = solve_triangular(L, Y_use, lower=True, check_finite=False)

    # OLS on whitened system
    AtA = A_white.T @ A_white
    if ridge:
        AtA = AtA + ridge * cp.eye(AtA.shape[0], dtype=AtA.dtype)
    AtY = A_white.T @ Y_white
    coeffs = cp.linalg.solve(AtA, AtY)

    # GLS pseudoinverse: (A'^T A')^{-1} A'^T L^{-1}
    # Compute A'^T L^{-1} without forming L^{-1}:
    # Let Z satisfy L^T Z = A_white  => Z = (L^T)^{-1} A_white, so Z^T = A_white^T L^{-1}
    Z = solve_triangular(L.T, A_white, lower=False, check_finite=False)   # (N_valid, P)
    A_pinv_cov = cp.linalg.solve(AtA, Z.T)                                # (P, N_valid)

    return coeffs, A_pinv_cov, mask