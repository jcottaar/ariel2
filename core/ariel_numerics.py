import numpy as np
import cupy as cp
import kaggle_support as kgs
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
        

def remove_trend_cp(x, window_size=10, degree=2):
    """
    Polynomial (Savitzky-Golay–like) detrend along axis 0 on GPU (CuPy).
    Works for arrays of any dimensionality. Returns residuals with the same
    shape as the input. Filtering is always applied over axis 0.

    Parameters
    ----------
    x : array_like
        Input data of shape (N, ...). Any dtype convertible to CuPy.
    window_size : int, default=10
        Length of the sliding window (must be > degree+1 and <= N).
    degree : int, default=2
        Polynomial degree used for local fit.

    Returns
    -------
    residuals : cupy.ndarray
        Same shape as `x`, containing x - local_polynomial_fit at each index
        along axis 0. Edges are handled via reflection so the length is preserved.

    Notes
    -----
    - The filter is *centered*. For even window sizes, the evaluation point is
      the left of the two center samples (index floor(w/2)).
    - Computation is batched across all remaining dimensions using a 2D view.
    """
    x = cp.asarray(x)
    if x.ndim < 1:
        raise ValueError("`x` must have at least 1 dimension (N, ...).")

    N = x.shape[0]
    w = int(window_size)
    p = int(degree) + 1

    if N < w:
        raise ValueError(f"Length along axis 0 ({int(N)}) is shorter than window size ({w}).")
    if w <= p:
        raise ValueError(f"Window size ({w}) must be > degree+1 ({p}).")

    # Flatten all non-time axes into one batch dimension to vectorize the fit.
    orig_shape = x.shape
    C = int(cp.prod(cp.asarray(orig_shape[1:])).item()) if x.ndim > 1 else 1
    x2 = x.reshape(N, C)

    # Centered padding so we get an output of length N after sliding.
    # Reflection padding reduces edge bias vs constant/zero padding.
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    xpad = cp.pad(x2, ((pad_left, pad_right), (0, 0)), mode='reflect')

    # Build sliding windows along axis 0: (N, C, w)
    windows = cp.lib.stride_tricks.sliding_window_view(xpad, w, axis=0)  # (N, C, w)

    # Vandermonde + pseudoinverse (shared across batches)
    xw = cp.arange(w, dtype=x.dtype)
    V = cp.vander(xw, p)              # (w, p) with powers [p-1 ... 0]
    pinv = cp.linalg.pinv(V)          # (p, w)

    # Evaluate the fitted polynomial at the (left) center position
    x0 = cp.asarray([w // 2], dtype=x.dtype)
    v0 = cp.vander(x0, p)[0]          # (p,)
    # Convolution-like weights to produce the fitted value at center from a window
    h = v0 @ pinv                     # (w,)

    # Predicted center value for every window and channel: (N, C)
    yhat_center = cp.tensordot(windows, h, axes=([2], [0]))

    # Residual at each position (length preserved): (N, C)
    residuals = x2 - yhat_center

    # Reshape back to original
    return residuals.reshape(orig_shape)


def estimate_noise_cp(ts, window_size=10, degree=2, combine_method='rms'):
    """
    Vectorized CuPy version.
    If `ts` is 1D (N,), returns a Python float.
    If `ts` is 2D (N, K), returns a 1D CuPy array of length K (one value per column).
    """
    ts = cp.asarray(ts)
    if ts.ndim == 1:
        ts = ts[:, None]
        squeeze_out = True
    elif ts.ndim == 2:
        squeeze_out = False
    else:
        raise ValueError("`ts` must be 1D or 2D (N,) or (N, K).")

    N, K = ts.shape
    w = window_size
    p = degree + 1
    if N < w:
        raise ValueError(f"Time series length ({int(N)}) is shorter than window size ({w}).")
    if w <= p:
        raise ValueError(f"Window size ({w}) must be > degree+1 ({p}).")

    # Sliding windows along time axis -> (num_windows, K, w)
    windows = cp.lib.stride_tricks.sliding_window_view(ts, w, axis=0)  # (N-w+1, K, w)

    # Vandermonde + pseudoinverse on GPU (shared across columns)
    x = cp.arange(w)
    V = cp.vander(x, p)          # (w, p)
    pinv = cp.linalg.pinv(V)     # (p, w)

    # Fit polynomial in each window for each column:
    # coeffs: (num_windows, K, p)
    coeffs = cp.tensordot(windows, pinv.T, axes=([2], [0]))
    # fitted: (num_windows, K, w)
    fitted = cp.tensordot(coeffs, V.T, axes=([2], [0]))

    # Residual RMS per window per column
    residuals = windows - fitted
    rms = cp.sqrt(cp.mean(residuals**2, axis=2))  # (num_windows, K)

    # Bias correction
    correction = cp.sqrt(w / (w - p))
    rms_corr = rms * correction  # (num_windows, K)

    # Combine across windows for each column
    if combine_method == 'rms':
        result = cp.sqrt(cp.mean(rms_corr**2, axis=0))   # (K,)
    elif combine_method == 'mean':
        result = cp.mean(rms_corr, axis=0)               # (K,)
    elif combine_method == 'median':
        result = cp.median(rms_corr, axis=0)             # (K,)
    else:
        raise ValueError("combine_method must be 'rms', 'mean' or 'median'")

    if squeeze_out:
        return result[0].item()
    return result

import cupy as cp

@kgs.profile_each_line
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

import numpy as np

@kgs.profile_each_line
def nan_pca_rank1_memmap(
    X_row: np.memmap,      # shape (R, C), order='C'  (fast row batches)
    X_col: np.memmap,      # shape (R, C), order='F'  (fast column batches)
    W_pre, C_pre, 
    max_iter: int = 50,
    tol: float = 1e-6,
    row_batch: int = 8192,   # tune to your RAM/SSD
    col_batch: int = 256,    # tune to your RAM/SSD
    verbose: bool = False,
    acc_dtype = np.float64   # accumulator dtype for stable sums
):
    """
    Rank-1 PCA with missing values using ALS, out-of-core.

    Model: X_ij ≈ W_i * C_j, for observed (non-NaN) entries.
    Updates:
      C_j = sum_i M_ij * W_i * X_ij / sum_i M_ij * W_i^2
      W_i = sum_j M_ij * C_j * X_ij / sum_j M_ij * C_j^2
    where M_ij = 1 if X_ij is observed else 0.

    Requirements:
      - X_row and X_col have identical shape (R, C), same dtype.
      - X_row is C-order memmap; X_col is F-order memmap.
      - Never copies the full array; only small row/column tiles are loaded.
    Returns:
      W: (R,) scores
      C: (1, C) component (row vector)
      S_post: (1,) strength (||W||_2)
    """
    if X_row.shape != X_col.shape:
        raise ValueError("X_row and X_col must have the same shape.")
    if X_row.dtype != X_col.dtype:
        raise ValueError("X_row and X_col must have the same dtype.")
    R, C = X_row.shape
    dtype = X_row.dtype

    def _nan_to_num(block):
        out = block.copy()
        np.nan_to_num(out, copy=False)
        return out

    # init C by column means
    C_vec = np.random.default_rng(seed=42).standard_normal(C, dtype=acc_dtype)

    W = np.zeros(R, dtype=acc_dtype)

    prev_C = C_vec.copy()
    for it in range(1, max_iter + 1):
        # plt.figure()
        # plt.imshow(C_vec.reshape(32,32), interpolation='none', aspect='auto')
        # plt.pause(0.001)
        C2 = (C_vec ** 2)
        for i0 in tqdm(range(0, R, row_batch)):
            #print(i0)
            i1 = min(i0 + row_batch, R)
            #num_i = np.zeros(i1 - i0, dtype=acc_dtype)
            #den_i = np.zeros(i1 - i0, dtype=acc_dtype)
            #for j0 in range(0, C, col_batch):
            #j1 = min(j0 + col_batch, C)
            j0 = 0
            j1 = C
            block = X_row[i0:i1, j0:j1] - W_pre[i0:i1,:]@C_pre[:, j0:j1]
            mask  = ~np.isnan(block)
            num_i = _nan_to_num(block) @ C_vec[j0:j1]
            den_i = (mask * C2[j0:j1]).sum(axis=1, dtype=acc_dtype)#mask @ C2[j0:j1]
            W[i0:i1] = np.divide(num_i, den_i, out=np.zeros_like(num_i), where=den_i > 0)
           # X_row._mmap.madvise(MADV_DONTNEED, start=i0*C*itemsize, length=(i1-i0)*C*itemsize)

        for j0 in tqdm(range(0, C, col_batch)):
            j1 = min(j0 + col_batch, C)
            #num = np.zeros(j1 - j0, dtype=acc_dtype)
            #den = np.zeros(j1 - j0, dtype=acc_dtype)
            #for i0 in range(0, R, row_batch):
            i0 = 0
            i1 = R
            block = X_col[i0:i1, j0:j1] - W_pre[i0:i1,:]@C_pre[:, j0:j1]
            num = _nan_to_num(block).T @ W[i0:i1]
            den = (~np.isnan(block)).T @ (W[i0:i1] ** 2)
            C_vec[j0:j1] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        cnorm = np.linalg.norm(C_vec)
        if cnorm > 0:
            C_vec /= cnorm
            W *= cnorm

        rel = np.linalg.norm(C_vec - prev_C) / (np.linalg.norm(prev_C) + 1e-12)
        if verbose:
            print(f"iter {it:02d}  ΔC_rel={rel:.3e}")
        if rel < tol:
            break
        prev_C[:] = C_vec

    W_out = W.astype(dtype, copy=False)
    C_out = C_vec.astype(dtype, copy=False)[None, :]
    S_post = np.array([np.linalg.norm(W_out)], dtype=dtype)
    return W_out, C_out, S_post

import cupy as cp

def lstsq_nanrows_normal_eq_with_pinv_sigma(
    dat,
    A,
    sigma = None,
    return_A_pinv_w = True
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
    if return_A_pinv_w:        
        A_pinv_w = cp.linalg.solve(AtA, (A_w.T * w[None, :]))  # (P, N_valid)
        return coeffs, A_pinv_w, mask
    else:
        return coeffs,mask

    

import cupy as cp
from cupyx.scipy.linalg import solve_triangular
