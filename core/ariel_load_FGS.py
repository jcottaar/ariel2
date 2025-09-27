'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

The code below deals with jitter correction for AIRS.
Unfortunately, I wasn't able to get this working properly. What I have here is better than naive binning, but not by much.
I am sure it is possible to do much better.
See my writeup for some indication of what I was going for.
'''
import numpy as np
import cupy as cp
import copy
import kaggle_support as kgs
import matplotlib.pyplot as plt
import ariel_numerics
import cupyx.scipy.sparse

import warnings
from scipy.sparse import SparseEfficiencyWarning  # works for cupyx too
warnings.simplefilter("ignore", SparseEfficiencyWarning)

diagnostic_plots = False

C0_combined = kgs.dill_load(kgs.calibration_dir + 'FGS_C0_2.pickle')
(cov,C_combined,Psi,Sigma_model) = kgs.dill_load(kgs.calibration_dir + 'FGS_jitter.pickle')
C_combined = C_combined.T
N_components_use = 9
design_matrix_combined = cp.concatenate((C0_combined,cp.array(C_combined)))
M=1024
rows = 1025 * cp.arange(M, dtype=cp.int64)             # 0, 1025, 2050, ...
cols = N_components_use + cp.arange(M, dtype=cp.int64) # offset columns
dataa = cp.ones(M, dtype=cp.float32)
add_block = cupyx.scipy.sparse.csc_matrix((dataa, (rows, cols)), shape=(1024*1024, N_components_use+1024))
FGS_weights = cp.array(kgs.dill_load(kgs.calibration_dir + '/FGS_weights.pickle'))[:,None]



class ApplyWavelengthBinningFGS2(kgs.BaseClass):
    
    def __call__(self, data, planet, observation_number):
        coeffs = get_coeffs(data.data)[0]
        res = cp.sum(FGS_weights*coeffs,0).reshape(-1,1)        
        data.data = res
        data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])

def get_coeffs(data):
    
    data = data.reshape(-1,1024)
    isnan = cp.isnan(data[0,:])

    # Estimate noise
    detrended = ariel_numerics.remove_trend_cp(data)[10:-10,...]
    detrended[:,isnan]=0
    rhs = detrended.T@detrended/(detrended.shape[0]-1)
    rhs=rhs.flatten()

    design_matrix_noise = cupyx.scipy.sparse.csc_matrix((1024*1024, N_components_use+1024))    
    design_matrix_part = cp.zeros((1024*1024,N_components_use))
    for ii in range(N_components_use):
        x = copy.deepcopy(design_matrix_combined[[ii],:])
        x[:,isnan]=0
        x1 = (x.T@x).flatten()        
        design_matrix_part[:,ii] = x1
    design_matrix_noise[:,:N_components_use] = design_matrix_part  
    design_matrix_noise += add_block    
    b = cp.array(design_matrix_noise.T.get() @ rhs.get())
    coeffs = cp.linalg.solve((design_matrix_noise.T@design_matrix_noise).todense(), b)
    noise_est = cp.sqrt(coeffs[N_components_use:])
    noise_est[cp.isnan(noise_est) | (noise_est<1)] = 1

    noise_est_naive = cp.sqrt(4+0.018*cp.abs(cp.mean(data,0)))
    to_change = noise_est<noise_est_naive
    noise_est[to_change] = noise_est_naive[to_change]
    
    # Fit
    design_matrix = design_matrix_combined            
    N = design_matrix.shape[0]
    res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(data.T, design_matrix.T, return_A_pinv_w=True, sigma=noise_est)
    coeffs = res[0]
    pred = design_matrix.T@coeffs
    residual = (data.T - pred).T
    
    return (coeffs,cp.sum(pred,0))