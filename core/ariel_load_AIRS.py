import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import copy
from dataclasses import dataclass, field, fields
import kaggle_support as kgs
import pyarrow.parquet
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import ariel_numerics
import ariel_load_FGS


AIRS_C0 = kgs.dill_load(kgs.calibration_dir + 'AIRS_C0_2.pickle')
AIRS_C = kgs.dill_load(kgs.calibration_dir + 'AIRS_jitter.pickle')[0]
AIRS_design_matrix = cp.concatenate([AIRS_C0.reshape(1,32,282), cp.array(AIRS_C[:2,:]).reshape(2,32,282)])
AIRS_design_matrix_np = AIRS_design_matrix.get()
del AIRS_C0; del AIRS_C;
print(AIRS_design_matrix_np.shape)

AIRS_C6 = kgs.dill_load(kgs.calibration_dir + 'AIRS_C6.pickle')
print(AIRS_C6.shape)
AIRS_design_matrix = AIRS_C6.transpose( (2,0,1) )
AIRS_design_matrix_np = AIRS_design_matrix.get()
print(AIRS_design_matrix_np.shape)
AIRS_weights6 = cp.array(kgs.dill_load(kgs.calibration_dir + 'AIRS_weights6.pickle'))

class ApplyWavelengthBinningAIRS3(kgs.BaseClass):
    residual_threshold = np.inf
    combine_rr2 = False
    cutoff_sum=0
    use_noise_est_naive = False
    sequential_fit = False
    handle_mean = True
    #alpha=-0.5
    
    # Diagnostics
    #residual = None
    
    n_iter = 3
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        assert not data.is_FGS
        
        dataa = data.data
        
        # Estimate noise
        # result = cp.empty((dataa.shape[0], dataa.shape[2]))
        # residual = cp.empty_like(dataa)
        # residual_expected = cp.zeros((32,282))
        # mean_vals = cp.zeros(282)
        noise_est_full = cp.zeros((32,282))
        #noise_est_naive = cp.zeros((32,282))
        noise_est_naive = 0.4*cp.sqrt(64+cp.abs(cp.mean(dataa,0)))
        # all_coeffs=[]
        
        for i_wavelength in range(282):
        
            isnan = cp.isnan(dataa[0,:,i_wavelength]).get()
            
            # Determine noise
            rhs = ariel_numerics.estimate_noise_cov_cp(dataa[:,:,i_wavelength]).get()
            rhs[isnan,:] = 0
            rhs[:,isnan] = 0        
            rhs=rhs.flatten()
            
            design_matrix = np.zeros((32*32, 3+32))
            for ii in range(3):
                x = AIRS_design_matrix_np[ii,:,[i_wavelength]]
                x[:,isnan]=0
                x1 = (x.T@x).flatten()
                design_matrix[:,ii] = x1

            for ii in range(32):
                design_matrix[33*ii,ii+3]=1
                
            coeffs = np.linalg.lstsq(design_matrix, rhs, rcond=None)[0]    
            residual_cov = rhs - design_matrix@coeffs       
            
            noise_est2 = cp.array(coeffs[3:])
            to_change = noise_est2<noise_est_naive[:,i_wavelength]**2
            noise_est2[to_change] = (noise_est_naive[:,i_wavelength]**2)[to_change]
            noise_est = cp.sqrt(noise_est2)
            noise_est[isnan] = 0
            assert not cp.any(cp.isnan(noise_est))
            
            noise_est_full[:,i_wavelength] = noise_est
            
            residual_cov_ratio = residual_cov/cp.max(rhs)
            #kgs.sanity_check(kgs.rms, residual_cov_ratio, 'residual_cov_rms', 1, [0,0.02])
            #kgs.sanity_check(lambda x:np.max(np.abs(x)), residual_cov_ratio, 'residual_cov_max', 2, [0,0.2])


        n_batch = dataa.shape[0]
        n_comp = AIRS_C6.shape[2]
        n_wavelength = AIRS_C6.shape[1]
        n_r = AIRS_C6.shape[0]
        C = AIRS_C6
        W = cp.zeros((n_batch,n_comp))
        W[:,0] = 1
        rhs = dataa.reshape(n_batch,n_r*n_wavelength,1)
        isnan = cp.isnan(rhs[0,:,0])           
        rhs[:,isnan,:] = 0

        # --- CHANGED: use noise_est for prewhitening ---
        noise_est = noise_est_full.flatten()
        inv_noise = cp.zeros_like(noise_est)
        inv_noise[~isnan] = 1.0 / noise_est[~isnan]
        inv_noise[isnan] = 0
        rhs_w = rhs * inv_noise[None, :, None]
        # ----------------------------------------------

        for ii in range(self.n_iter):

            # Fit W0
            sum_wi_ci = cp.zeros((n_batch, n_r, n_wavelength))
            for i_comp in range(n_comp):
                sum_wi_ci += W[:,None,None,i_comp] * C[None,:,:,i_comp]
            sum_wi_ci = sum_wi_ci.reshape(n_batch, n_r*n_wavelength)
            sum_wi_ci[:,isnan] = 0 
            sum_wi_ci = sum_wi_ci.reshape(n_batch, n_r, n_wavelength)
            rhs2 = rhs.reshape(n_batch, n_r, n_wavelength)

            # --- CHANGED: weighted W0 ---
            w2 = inv_noise.reshape(n_r, n_wavelength)
            rhs2_w = rhs2 * w2[None, :, :]
            sum_wi_ci_w = sum_wi_ci * w2[None, :, :]
            W0 = cp.sum(rhs2_w * sum_wi_ci_w, 1) / cp.sum(sum_wi_ci_w * sum_wi_ci_w, 1)
            # ----------------------------

            # Fit W
            design_matrix = W0[:,None,:,None]*C[None,:,:,:]
            design_matrix = design_matrix.reshape(n_batch, n_r*n_wavelength,n_comp)            
            design_matrix[:,isnan,:]=0

            # --- CHANGED: prewhiten design and rhs for GLS ---
            design_matrix_w = design_matrix * inv_noise[None, :, None]
            AtA = cp.einsum('tmn,tmk->tnk', design_matrix_w, design_matrix_w)
            Atb = cp.einsum('tmn,tmk->tnk', design_matrix_w, rhs_w)
            # -------------------------------------------------

            coeffs = cp.linalg.solve(AtA,Atb)
            assert not cp.any(cp.isnan(coeffs))
            W = coeffs[:,:,0]
            
        W_scaled = cp.sum(W*AIRS_weights6[None,:],1)
        result = W0 * W_scaled[:,None]
        result = result*cp.sum(AIRS_C6[:,:,0],0)
        data.data = result
        data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])
           
            
        
        