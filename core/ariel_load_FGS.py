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

#print(FGS_weights)


class ApplyWavelengthBinningFGS2(kgs.BaseClass):
    
    n_mean_pixels = 0
    
    def __call__(self, data, planet, observation_number):
        coeffs = get_coeffs(data.data, n_mean_pixels=self.n_mean_pixels)[0]
        res = cp.sum(FGS_weights*coeffs,0).reshape(-1,1)
        planet.transits[observation_number].diagnostics['FGS_jitter']=coeffs[3,:].get()
        if diagnostic_plots:
            plt.figure()
            plt.plot(coeffs[3,:].get())
            plt.pause(0.001)
        data.data = res
        data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])

def get_coeffs(data, n_mean_pixels=-1):
    
    data = data.reshape(-1,1024)
        
    isnan = cp.isnan(data[0,:])
    #isnan = cp.zeros_like(isnan)

    detrended = ariel_numerics.remove_trend_cp(data)[10:-10,...]
    detrended[:,isnan]=0
    rhs = detrended.T@detrended/(detrended.shape[0]-1)
    #print(rhs)
    # plt.figure()
    # plt.imshow(detrended[0,...].get().reshape(32,32))
    # plt.colorbar()
    #rhs = ariel_numerics.estimate_noise_cov_cp(data)
    #rhs = cp.array(cov)
    
#     plt.figure()
#     plt.imshow(rhs[400:600,400:600].get())
#     plt.colorbar()
    
#     plt.figure()
#     plt.imshow(cov[400:600,400:600])
#     plt.colorbar()
    
#     plt.figure()
#     plt.imshow(rhs[400:600,400:600].get())
#     plt.colorbar()
    
#     plt.figure()
#     plt.imshow(rhs[400:600,400:600].get()/cov[400:600,400:600])
#     plt.colorbar()

    #rhs[isnan,:] = 0
    #rhs[:,isnan] = 0

    rhs=rhs.flatten()
    
    # plt.figure()
    # plt.plot(rhs.get())
    
    design_matrix_noise = cupyx.scipy.sparse.csc_matrix((1024*1024, N_components_use+1024))    
    design_matrix_part = cp.zeros((1024*1024,N_components_use))
    for ii in range(N_components_use):
        x = copy.deepcopy(design_matrix_combined[[ii],:])
        x[:,isnan]=0
        x1 = (x.T@x).flatten()        
        design_matrix_part[:,ii] = x1
    design_matrix_noise[:,:N_components_use] = design_matrix_part

    # --- vectorized construction of the second-loop entries ---    
    design_matrix_noise += add_block
       # for ii in range(1024):
       #     design_matrix_noise[1025*ii,ii+N_components_use]=1

    #plt.figure()
    #plt.plot(design_matrix_noise.sum(0).T.get())

    #print((design_matrix_noise.T@design_matrix_noise).shape)
    
    #b = ariel_numerics.spmv_csc_T_times_vec_deterministic(design_matrix_noise.T, rhs)
    b = cp.array(design_matrix_noise.T.get() @ rhs.get())
    #print(kgs.rms((design_matrix_noise.T@design_matrix_noise).todense().get()), kgs.rms(rhs.get()), kgs.rms(b.get()))
    coeffs = cp.linalg.solve((design_matrix_noise.T@design_matrix_noise).todense(), b)
    #print(coeffs.flatten()[0])
    #residual = rhs - design_matrix_noise*coeffs
    noise_est = cp.sqrt(coeffs[N_components_use:])
    noise_est[cp.isnan(noise_est) | (noise_est<1)] = 1
    
    # plt.figure()
    # plt.scatter(cp.sqrt(cp.abs(cp.mean(data,0))).get(), noise_est.get())
    # x=cp.sqrt(cp.abs(cp.mean(data,0))).get()
    # plt.scatter(x, np.sqrt(4+0.018*x**2))
    #plt.xlim([0,10])
    #plt.ylim([0,10])
    
#     plt.figure()
#     plt.scatter(2+cp.sqrt(cp.abs(cp.mean(data,0))).get(), noise_est.get())
    
# #     print(np.min(noise_est), np.max(noise_est), np.sum(np.isnan(noise_est)))
    
#     noise_est_naive = 2+cp.sqrt(design_matrix_combined[0,:])
    
#     plt.figure()
#     plt.scatter(noise_est_naive.get(), noise_est.get())
    
        
#     plt.figure()
#     plt.scatter(cp.sqrt(cp.abs(cp.mean(data,0))).get(), np.sqrt(np.diag(Psi)))
    
    # plt.figure()
    # plt.scatter(np.sqrt(np.diag(Psi)), noise_est.get(), )
    # plt.axline((0,0),slope=1/0.6, color='black')
    
#     plt.figure()
#     plt.imshow(noise_est.get().reshape(32,32))
#     plt.colorbar()
    
#     plt.figure()
#     plt.imshow(noise_est.get().reshape(32,32)/np.sqrt(np.diag(Psi)).reshape(32,32))
#     plt.colorbar()
    
    noise_est_naive = cp.sqrt(4+0.018*cp.abs(cp.mean(data,0)))
    to_change = noise_est<noise_est_naive
    #print(cp.sum(to_change))
    noise_est[to_change] = noise_est_naive[to_change]
    
    # plt.figure()
    # plt.scatter(cp.sqrt(cp.abs(cp.mean(data,0))).get(), noise_est.get())
    # x=cp.sqrt(cp.abs(cp.mean(data,0))).get()
    #plt.scatter(x, np.sqrt(4+0.018*x**2))
    
    # plt.figure()
    # plt.scatter(np.sqrt(np.diag(Psi)), noise_est.get(), )
    # plt.axline((0,0),slope=1/0.6, color='black')

    #print(noise_est.flatten()[0])
      

    if n_mean_pixels!=0:
        mean_handled = False
    else:
        mean_handled = True
    while True:

        if mean_handled:
            design_matrix = design_matrix_combined            
        else:
            if n_mean_pixels==-1:
                design_matrix = cp.concatenate((design_matrix_combined, cp.ones((1,1024))))
            else:
                design_matrix = design_matrix_combined[:1,:]

        N = design_matrix.shape[0]

        res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(data.T, design_matrix.T, return_A_pinv_w=True, sigma=noise_est)
        coeffs = res[0]
        
        #print(coeffs.flatten()[0])
        
        #print((design_matrix@design_matrix.T).shape)
        # plt.figure()
        # plt.imshow(res[1].get())
        # plt.colorbar()
        # plt.title('res[1]')
        # plt.figure()
        # plt.imshow((design_matrix@design_matrix.T).get())
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(cp.log(cp.linalg.inv(design_matrix@design_matrix.T)).get())
        # plt.colorbar()
        # plt.figure()
        # plt.imshow((cp.log(design_matrix).get()), aspect='auto', interpolation='none')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(data.get())
        # plt.figure()
        # plt.plot(C0_combined.T.get())
        # plt.figure()
        # plt.plot(design_matrix[0,:].T.get())
        
        # plt.figure()
        # X = coeffs.T
        # X = X-cp.mean(X,0)
        # plt.plot(X.get())
        
        pred = design_matrix.T@coeffs
        residual = (data.T - pred).T

#         residualf[:,:,i_wavelength] = (data[:,:,i_wavelength].T - design_matrix.T@coeffs).T           #

#         #print(res[1].shape, cp.diag(noise_est**2).shape)
#         A_pinv_w = res[1]
#         A_pinv_w_full = cp.zeros((N,32))
#         A_pinv_w_full[:,~cp.isnan(data[0,:,i_wavelength])] = A_pinv_w
#         noise_est[cp.isnan(data[0,:,i_wavelength])] = 0
#         noise_output[i_wavelength] = cp.sqrt((A_pinv_w_full@cp.diag(noise_est**2)@A_pinv_w_full.T)[0,0])

#         noise_seen[i_wavelength] = ariel_numerics.estimate_noise_cp(coeffs[0,:])

# #             res2 = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(data[:,:,i_wavelength].T, design_matrix[0:3,:].T, return_A_pinv_w=True, sigma=noise_est)

# #             A_pinv_w = res2[1]
# #             A_pinv_w_full = cp.zeros((3,32))
# #             A_pinv_w_full[:,~cp.isnan(data[0,:,i_wavelength])] = A_pinv_w
# #             noise_est[cp.isnan(data[0,:,i_wavelength])] = 0
# #             noise_no_jitter[i_wavelength] = cp.sqrt((A_pinv_w_full@cp.diag(noise_est**2)@A_pinv_w_full.T)[0,0])

#         mat = design_matrix.T@A_pinv_w_full
#         #print(mat.shape)
#         cov_expected = cp.diag(noise_est**2) - mat@cp.diag(noise_est**2)@mat.T
#         residual_expected[:,i_wavelength] = cp.sqrt(cp.diag(cov_expected))

#         residual_expected_ratio = cp.mean(residualf[:,:,i_wavelength],0)/residual_expected[:,i_wavelength]*np.sqrt(1350)
#         residual_expected_ratio[cp.isnan(residual_expected_ratio)] = 0

#         #if i_wavelength>200 and i_wavelength<220:
#         #    print(i_wavelength, residual_expected_ratio, cp.max(cp.abs(residual_expected_ratio)))
#         # if cp.max(cp.abs(residual_expected_ratio))==0:
        #     plt.figure()
        #     plt.imshow(cov_expected.get())
        #     plt.figure()
        #     #plt.plot(residual_expected[:,i_wavelength].get())
        #     plt.plot(cp.std(residualf[:,:,i_wavelength].get()))
        #     print(cp.std(residualf[:,:,i_wavelength].get()))
        #     raise 'stop'
        if False:#cp.any(cp.abs(residual_expected_ratio)>30):
            #print(i_wavelength, cp.max(cp.abs(residual_expected_ratio)), cp.argmax(cp.abs(residual_expected_ratio)), mean_handled)
            data[:,cp.argmax(cp.abs(residual_expected_ratio)),i_wavelength] = cp.nan
        else:
            #mean_handled = True
            if not mean_handled:
                #print(coeffs[3,:].shape)
                if n_mean_pixels==-1:
                    mean_estimate = np.mean(coeffs[-1,:])    
                else:
                    mean_per_pixel = cp.mean(residual, axis=0).flatten()
                    inds = cp.argsort(cp.mean(data,axis=0).flatten())
                    mean_estimate = cp.mean(cp.sort(mean_per_pixel[inds[:n_mean_pixels]]))
                data-=mean_estimate
                if diagnostic_plots:
                    print(mean_estimate)
                mean_handled = True
            else:
                break

    if diagnostic_plots:
        plt.figure()
        plt.imshow((cp.mean(residual,0)).reshape(32,32).get())
        plt.title(cp.nanmean(residual).get())
        plt.colorbar()
        pred2 = design_matrix[:1,:].T@coeffs[:1,:]
        residual2 = (data.T - pred2).T
        plt.figure()
        plt.imshow((cp.mean(residual2,0)).reshape(32,32).get())
        plt.title(cp.nanmean(residual2).get())
        plt.colorbar()
        mean_per_pixel = cp.mean(residual, axis=0).flatten()
        #plt.figure();plt.semilogy(cp.sort(mean_per_pixel).get())
        inds = cp.argsort(cp.mean(data,axis=0).flatten())
        print(cp.mean(cp.sort(mean_per_pixel[inds[:100]])))
        mean_per_pixel = cp.mean(residual2, axis=0).flatten()
        #plt.figure();plt.semilogy(cp.sort(mean_per_pixel).get())
        inds = cp.argsort(cp.mean(data,axis=0).flatten())
        print(cp.mean(cp.sort(mean_per_pixel[inds[:100]])))
        
    
#     plt.figure()
#     plt.imshow(cp.std(residual,0).reshape(32,32).get())
#     plt.colorbar()
    
#     plt.figure()
#     plt.imshow((cp.mean(residual,0)/(cp.std(residual,0))).reshape(32,32).get())
#     plt.colorbar()

    #print(coeffs.flatten()[0])

    return (coeffs,cp.sum(pred,0))