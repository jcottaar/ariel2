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