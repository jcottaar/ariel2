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

diagnostic_plots = False

#@kgs.profile_each_line
def read_parquet_to_numpy(filename):
    table = pyarrow.parquet.read_table(filename)
    data = table.to_pandas().to_numpy()
    return data

def read_parquet_to_cupy(filename):
    return cp.array(read_parquet_to_numpy(filename))

def inpaint_along_axis_inplace(arr, axis=0):
    arr = cp.asarray(arr)
    ndim = arr.ndim
    T = arr.shape[axis]

    # Move the interpolation axis to the front
    arr_moved = cp.moveaxis(arr, axis, 0)  # shape (T, ...)
    flat_arr = arr_moved.reshape(T, -1)    # shape (T, N)

    nan_mask = cp.isnan(flat_arr)
    needs_interp = cp.any(nan_mask, axis=0)

    x = cp.arange(T)

    for i in cp.where(needs_interp)[0]:
        col = flat_arr[:, i]
        valid_mask = ~cp.isnan(col)
        valid_x = x[valid_mask]
        valid_y = col[valid_mask]

        if valid_x.size == 0:
            flat_arr[:, i] = cp.nan
        else:
            flat_arr[:, i] = cp.interp(x, valid_x, valid_y)

    # No need to move axes back — arr was modified in-place through views
    
pca_data = kgs.dill_load(kgs.calibration_dir + '/explore_bad_pixels_pca.pickle')
coeff_data = kgs.dill_load(kgs.calibration_dir+'/explore_base_shape_from_pca_coeff_list.pickle')
core_shapes = []
for i_wavelength in range(283):
    core_shapes.append((cp.stack([c[0,:] for c in pca_data[1][i_wavelength]]).T @ coeff_data[i_wavelength][2][:,None])[:,0])
    

@dataclass
class apply_pca_modelOptions(kgs.BaseClass):
    n_components: int = field(init=True, default=0)
    use_sum: bool = field(init=True, default=False)
    
    include_diagnostics: bool = field(init=True, default=False)

def apply_pca_model(data, wavelength_ids, options, residual_mode = 0):
    # 0: no residual
    # 1: standard residual
    # 2: remove only main shape
    options.check_constraints()
    
    if residual_mode>0:
        residual = copy.deepcopy(data)
    else:
        residual = None
    weighted_coeffs = cp.empty((data.shape[0], data.shape[2]))
    for i_data, i_wavelength in enumerate(wavelength_ids):
        this_components = pca_data[1][i_wavelength][:options.n_components]
        this_data = data[:,:,i_data]
        noise_est = cp.sqrt(cp.abs(this_components[0]))[0,:]        
        design_matrix = cp.stack([c[0,:] for c in this_components]).T                                                            
        res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(this_data.T, design_matrix, sigma=noise_est, return_A_pinv_w=options.include_diagnostics)
        coeffs = res[0].T      
        if not options.use_sum:
            weighted_coeffs[:,i_data] = coeffs @ coeff_data[i_wavelength][1][options.n_components-1]
        else:
            weighted_coeffs[:,i_data] = cp.sum((design_matrix@coeffs.T).T,1)
        if residual_mode == 1:
            residual[:,:,i_data] = (this_data.T-design_matrix@coeffs.T).T
        elif residual_mode == 2:
            residual[:,:,i_data] = this_data - weighted_coeffs[:,i_data][:,None] * core_shapes[i_wavelength]
        
        #print(coeffs.shape, this_data.shape, residual.shape)
        
    output = (weighted_coeffs, residual)
    return output

#@kgs.profile_each_line
def inpaint_vectorized(data):
    """
    Inpaints NaN‐patches along the last axis (Y) of `data` (shape C×X×Y)
    by:
      • flat‐extrapolating any NaN runs at the left or right edge
      • linearly interpolating any interior NaN runs
    Loops only over X and over NaN‐segments (rare), otherwise fully GPU‐vectorized.
    """
    C, X, Y = data.shape

    for x in range(X):
        # 1) find the mask of invalids along Y for this column
        mask = cp.isnan(data[0, x, :])
        if not mask.any():
            continue  # no NaNs here

        # if there are absolutely no valid pixels, skip (or fill with zeros if you prefer)
        if not (~mask).any():
            continue

        # 2) find where mask flips -> segment starts/ends
        diff = cp.diff(mask.astype(cp.int8))  # +1: False→True,  -1: True→False
        starts = cp.where(diff == 1)[0] + 1    # indices where a NaN‐run begins
        ends   = cp.where(diff == -1)[0]       # indices where a NaN‐run ends

        # if the very first pixel is NaN, that run really starts at 0
        if mask[0]:
            starts = cp.concatenate((cp.array([0], dtype=starts.dtype), starts))
        # if the very last pixel is NaN, that run ends at Y–1
        if mask[-1]:
            ends = cp.concatenate((ends, cp.array([Y-1], dtype=ends.dtype)))

        # 3) fill each [start … end] segment
        for st, en in zip(starts, ends):
            # left‐edge run?
            if st == 0:
                # flat‐fill with first valid to the right
                right = en + 1
                fill = data[:, x, right][:, None]
                data[:, x, : right] = fill

            # right‐edge run?
            elif en == Y - 1:
                # flat‐fill with last valid to the left
                left = st - 1
                fill = data[:, x, left][:, None]
                data[:, x, left + 1 :] = fill

            # interior run: do linear interpolation
            else:
                left, right = st - 1, en + 1
                length = right - left
                # weights for positions y = left+1 … right-1
                ys = cp.arange(left.get() + 1, right.get())
                t = (ys - left).astype(data.dtype) / length
                wL, wR = 1 - t, t

                V_L = data[:, x, left][:, None]
                V_R = data[:, x, right][:, None]

                # broadcast over channels
                data[:, x, left + 1 : right] = wL[None, :] * V_L + wR[None, :] * V_R

   # final sanity check
    assert not cp.any(cp.isnan(data)), "Some NaNs remain after inpainting!"

    
    
def bin_first_axis(arr: cp.ndarray, bin_size: int) -> cp.ndarray:
    """
    Bin `arr` along axis=0 in groups of `bin_size`, averaging within each bin.
    Drops any leftover rows at the end if arr.shape[0] % bin_size != 0.
    
    Parameters
    ----------
    arr : cupy.ndarray
        Input array of shape (N, D1, D2, ..., Dk).
    bin_size : int
        Number of consecutive slices along axis-0 to average together.
        
    Returns
    -------
    binned : cupy.ndarray
        Array of shape (N//bin_size, D1, D2, ..., Dk), where each [i] is the
        mean of arr[i*bin_size:(i+1)*bin_size, ...].
    """
    N = arr.shape[0]
    n_bins = N // bin_size
    if n_bins == 0:
        raise ValueError(f"bin_size={bin_size} too large for array length {N}")
    
    # trim off the tail
    trimmed = arr[: n_bins * bin_size]
    
    # reshape & average
    new_shape = (n_bins, bin_size) + trimmed.shape[1:]
    return trimmed.reshape(new_shape).mean(axis=1)

#pca_data = kgs.dill_load(kgs.temp_dir + '/explore_bad_pixels_pca.pickle')[1]

AIRS_jitter = cp.array(kgs.dill_load(kgs.calibration_dir + 'AIRS_jitter.pickle')[0][:2,:], dtype=cp.float64)
AIRS_base_scaling = cp.array(kgs.dill_load(kgs.calibration_dir + 'AIRS_base_scaling.pickle'))


@dataclass
class LoadRawData(kgs.BaseClass):   
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        filename = planet.get_directory() + kgs.sensor_names[not data.is_FGS] + '_signal_' + str(observation_number) + '.parquet'
        data.data = read_parquet_to_cupy(filename)
        if kgs.profiling: cp.cuda.Device().synchronize()
        if data.is_FGS:
            data.data = cp.reshape(data.data, (135000, 32, 32))
        else:
            data.data = cp.reshape(data.data, (11250, 32, 356))                  
        if kgs.profiling: cp.cuda.Device().synchronize()

        # Times
        data.times = cp.array(kgs.axis_info[kgs.sensor_names[not data.is_FGS]+'-axis0-h'].to_numpy())*3600
        if kgs.profiling: cp.cuda.Device().synchronize()
        data.times = data.times[~cp.isnan(data.times)]
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Integration times
        if not data.is_FGS:
            dt = cp.array(kgs.axis_info['AIRS-CH0-integration_time'].dropna().values)
        else:
            dt = cp.ones(data.data.shape[0])*0.1
        dt[1::2] += 0.1
        data.time_intervals = dt
        
        # Wavelength
        if data.is_FGS:
            data.wavelengths = cp.array([kgs.wavelengths[0]])
        else:
            data.wavelengths = cp.array(kgs.axis_info['AIRS-CH0-axis2-um'].dropna().to_numpy())
                
        
        if kgs.profiling: cp.cuda.Device().synchronize()
     
@dataclass
class ApplyPixelCorrections(kgs.BaseClass):
    
    clip_columns = False
    clip_columns1 = None
    clip_columns2 = None
    
    clip_rows = False
    clip_rows1 = None
    clip_rows2 = None
    
    mask_dead = True
    mask_hot = False
    hot_sigma_clip = 5
    linear_correction = True
    dark_current = True
    flat_field = True
    remove_cosmic_rays = True
    cosmic_ray_threshold = 20
    remove_first_and_last_frame = False
    
    adc_offset_sign = 1 
    dark_current_sign = 1
    
    poke_holes = False
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        calibration_directory = planet.get_directory() + kgs.sensor_names[not data.is_FGS] + '_calibration_' + str(observation_number) + '/'
        
        def clip_3d(signal):     
            if self.clip_columns:
                signal = signal[:, :, self.clip_columns1:self.clip_columns2]
            if self.clip_rows:
                signal = signal[:, self.clip_rows1:self.clip_rows1, :]   
            return signal
                
        def clip_2d(signal):
            if self.clip_columns:
                signal = signal[:, self.clip_columns1:self.clip_columns2]
            if self.clip_rows:
                signal = signal[self.clip_rows1:self.clip_rows1, :] 
            return signal
        
        def ADC_convert(signal, gain, offset):
            signal /= gain
            signal += self.adc_offset_sign * offset
            kgs.sanity_check(lambda x:x, gain, 'gain', 1, [0.4, 0.5])                
            kgs.sanity_check(lambda x:x, offset, 'offset', 2, [-1001, -999])
            return signal

        def mask_hot_dead(signal, dead, dark):
            hot = cp.array(sigma_clip(dark.get(), sigma=self.hot_sigma_clip, maxiters=5).mask)
            if kgs.sanity_checks_active:
                kgs.sanity_check(lambda x:x, cp.mean(hot), 'ratio_hot', 3, [0, 0.018])  # ~0.0166 probed in test set
                kgs.sanity_check(lambda x:x, cp.mean(dead), 'ratio_dead', 4, [0, 0.012]) # ~0.0107 probed in test set
            if self.mask_hot:                
                signal[:, hot] = cp.nan
            else:
                pass
                #print('deal with extremely negative dark in test set first...also more sanity checks may fail')
            if self.mask_dead:
                signal[:, dead] = cp.nan
            return signal

        def apply_linear_corr(lc,x):        
            result =  lc[0,:,:]+x*(lc[1,:,:]+x*(lc[2,:,:]+x*(lc[3,:,:]+x*(lc[4,:,:]+x*lc[5,:,:]))))
            if kgs.sanity_checks_active:
                kgs.sanity_check(cp.nanmax, cp.abs(result-x), 'linear_corr_impact', 9, [0, 50000])
            return result

        def clean_dark(signal, dark, dt):    
            signal -= self.dark_current_sign * dark * dt[:, cp.newaxis, cp.newaxis]
            # dark_plot = copy.deepcopy(dark)
            # dark_plot[cp.isnan(signal[0,...])] = cp.nan
            # plt.figure()
            # plt.imshow(dark_plot.get(), aspect='auto', interpolation='none')
            # plt.colorbar()
            if self.mask_dead:
                if self.mask_hot:
                    kgs.sanity_check(cp.min, dark[~cp.isnan(signal[0,:,:])], 'dark_min', 5, [-0.01, 0.01]) 
                    kgs.sanity_check(cp.max, dark[~cp.isnan(signal[0,:,:])], 'dark_max', 6, [0.005, 0.02])        
                else:
                    kgs.sanity_check(cp.min, dark[~cp.isnan(signal[0,:,:])], 'dark_min', 5, [-np.inf-0.05, 0.01]) 
                    kgs.sanity_check(cp.max, dark[~cp.isnan(signal[0,:,:])], 'dark_max', 6, [0., 25.]) 
            return signal

        def correct_flat_field(flat, signal):        
            signal = signal / flat[cp.newaxis, :,:]
            if self.mask_dead:
                if self.mask_hot:
                    kgs.sanity_check(cp.min, flat[~cp.isnan(signal[0,:,:])], 'flat_min', 7, [0.5, 1.05]) # ~0.574 in test set
                    kgs.sanity_check(cp.max, flat[~cp.isnan(signal[0,:,:])], 'flat_max', 8, [0.95, 1.2])  
                else:
                    kgs.sanity_check(cp.min, flat[~cp.isnan(signal[0,:,:])], 'flat_min', 7, [0.5, 1.1])  # ~0.574 in test set
                    kgs.sanity_check(cp.max, flat[~cp.isnan(signal[0,:,:])], 'flat_max', 8, [0.9, 1.2]) 
            return signal
        
        def remove_cosmic_rays(signal):
            #diff = np.abs(signal[1:,:,:] - signal[:-1,:,:])
            #is_cosmic_ray = cp.empty_like(signal, dtype=bool)
            #is_cosmic_ray[0,...] = diff[0,...]>self.cosmic_ray_threshold
            #is_cosmic_ray[-1,...] = diff[-1,...]>self.cosmic_ray_threshold
            #is_cosmic_ray[1:-1,...] = ( (diff[1:,...]>self.cosmic_ray_threshold) & (diff[:-1,...]>self.cosmic_ray_threshold))
            is_cosmic_ray = cp.abs(signal - cp.mean(signal,0))/cp.std(signal,0) > self.cosmic_ray_threshold
            kgs.sanity_check(lambda x:cp.max(cp.mean(x)), is_cosmic_ray, 'cosmic_ray_removal', 10, [0,5e-6])
            signal[is_cosmic_ray] = cp.nan
            return signal
            
        
        # Load calibration data
        dark = clip_2d(read_parquet_to_cupy(calibration_directory+'dark.parquet'))
        dead = clip_2d(read_parquet_to_cupy(calibration_directory+'dead.parquet'))
        flat = clip_2d(read_parquet_to_cupy(calibration_directory+'flat.parquet'))
        linear_corr = clip_3d(read_parquet_to_cupy(calibration_directory+'linear_corr.parquet').reshape(6,32,-1))        
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Clip to desired rows and columns, and make float64
        data.data = clip_3d(data.data)
        data.data = data.data.astype(cp.float64)        
        if not data.is_FGS and self.clip_columns:
            data.wavelengths = data.wavelengths[self.clip_columns1:self.clip_columns2]
        if kgs.profiling: cp.cuda.Device().synchronize
        
        # ADC
        offset = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_offset'].to_numpy()[0]
        gain = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_gain'].to_numpy()[0]
        data.data = ADC_convert(data.data,gain,offset)
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Mask
        data.data = mask_hot_dead(data.data,dead>0,dark)
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Linear correction
        if self.linear_correction:
            data.data = apply_linear_corr(linear_corr, data.data)
        if kgs.profiling: cp.cuda.Device().synchronize()
            
        # Dark current
        if self.dark_current:            
            data.data = clean_dark(data.data,dark,data.time_intervals)            
        if kgs.profiling: cp.cuda.Device().synchronize()
            
        # Flat field
        if self.flat_field:
            data.data = correct_flat_field(flat, data.data)
        if kgs.profiling: cp.cuda.Device().synchronize()     
                 
        # Correlated double sampling    
        data.data = data.data[1::2,:,:]-data.data[0::2,:,:]
        data.times = data.times[0::2]/2+data.times[1::2]/2   
        data.time_intervals = data.time_intervals[1::2]
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Remove cosmic rays
        if self.remove_cosmic_rays:
            data.data = remove_cosmic_rays(data.data)
            inpaint_along_axis_inplace(data.data,0)
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        # Flip AIRS to have ascending wavelengths
        if not data.is_FGS:
            data.data = cp.flip(data.data, axis=2)
            data.data = cp.ascontiguousarray(data.data)
            data.wavelengths = cp.flip(data.wavelengths)
            
        if self.remove_first_and_last_frame:
            data.data = data.data[1:-1,...]
            data.times = data.times[1:-1]
            data.time_intervals = data.time_intervals[1:-1]
            
        if self.poke_holes:
            if data.is_FGS:
                data.data[:,15:16,15:16] = cp.nan
            else:
                data.data[:,15,::3] = cp.nan
                data.data[:,16,1::3] = cp.nan

@dataclass
class ApplyFullSensorCorrections(kgs.BaseClass):
        
    inpainting_time = True
    inpainting_wavelength = False
    inpainting_2d = False
    restore_invalids = False # after mean removal
    
    #remove_constant = 0.
    
    use_pca_for_background_removal = True
    pca_options: object = field(init=True, default_factory = lambda:apply_pca_modelOptions(n_components=4))
    
    remove_background_based_on_rows = False
    remove_background_n_rows = 8 
    remove_background_remove_used_rows = False
    
    remove_background_based_on_pixels = False
    remove_background_pixels = 100
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):        
        assert self.inpainting_time # actually done above
        was_invalid = cp.isnan(data.data[0,...])
        #if self.inpainting_time:
        #    inpaint_along_axis_inplace(data.data,0)
        if self.inpainting_wavelength:
            inpaint_vectorized(data.data)
        if self.inpainting_2d:     
            temp = cp.transpose(copy.deepcopy(data.data), (0,2,1))
            inpaint_vectorized(data.data)
            inpaint_vectorized(temp)
            data.data = (data.data+cp.transpose(temp, (0,2,1)))/2
        #data.data -= self.remove_constant
        
        #if self.remove_background_based_on_rows:
        
        if self.use_pca_for_background_removal:
            if data.is_FGS:
                data_pca = cp.reshape(data.data, (-1,1024,1))
                wavelength_ids = [0]
            else:
                data_pca = data.data
                wavelength_ids = np.arange(1,283)
            data_for_background_removal = apply_pca_model(data_pca, wavelength_ids, self.pca_options, residual_mode=2)[1]  
            data_for_background_removal = cp.reshape(data_for_background_removal, data.data.shape)    
            # if data.is_FGS:
            #     lims = [-1,5]
            # else:
            #     lims = [-1,50]
            # plt.figure()
            # plt.imshow(cp.mean(data.data,0).get(), aspect='auto', interpolation='none')
            # plt.clim(lims)
            # plt.colorbar()            
            # plt.figure()
            # plt.imshow(cp.mean(data_for_background_removal,0).get(), aspect='auto', interpolation='none')
            # plt.clim(lims)
            # plt.colorbar()            
            # plt.figure()
            # plt.imshow(cp.mean(data_for_background_removal-data.data,0).get(), aspect='auto', interpolation='none')
            # plt.clim([-1,1])
            # plt.colorbar()
            
        else:
            data_for_background_removal = data.data
        
        
        if data.is_FGS:
            # plt.figure()
            # plt.scatter(cp.sqrt(core_shapes[0]).get(), ariel_numerics.estimate_noise_cp(data.data.reshape(-1,1024)).get())
            # plt.figure()
            # plt.imshow(cp.mean(data.data,0).get(), aspect='auto', interpolation='none')
            #plt.figure()
            #plt.imshow(core_shapes[0].get().reshape(32,32))
            #plt.figure()
            #plt.imshow(cp.log(ariel_numerics.estimate_noise_cp(data.data.reshape(-1,1024))).get().reshape(32,32))
            pass
        else:
            pass
#             rr = apply_pca_model(cp.mean(data.data,0)[None,...], np.arange(1,283), self.pca_options, residual_mode=0)[0]
#             print(rr.shape)
            
#             noise_est = copy.deepcopy(data.data[0,:,:])
#             shp = copy.deepcopy(data.data[0,:,:])
#             for ii in range(282):
#                 noise_est[:,ii] = ariel_numerics.estimate_noise_cp(data.data[:,:,ii])#/cp.sqrt(rr[0,ii])
#                 shp[:,ii] = core_shapes[ii+1]*cp.sqrt(rr[0,ii])
# #              plt.figure()
#             # plt.imshow(cp.mean(data.data,0).get(), aspect='auto', interpolation='none')
#             # plt.figure()
#             # plt.imshow(shp.get(), aspect='auto', interpolation='none')
#             plt.figure()
#             plt.imshow(cp.log(noise_est).get(), aspect='auto', interpolation='none')
            # plt.figure()
            # for ii in range(32):
            #     plt.scatter(cp.sqrt(shp[ii,:]).get(), noise_est[ii,:].get())
        
        if not data.is_FGS:
            background_data = cp.concatenate((data_for_background_removal[:,:self.remove_background_n_rows,:], data_for_background_removal[:,-self.remove_background_n_rows:,:]), axis=1)
            background_estimate = cp.mean(background_data, axis=(0,1))
            if self.remove_background_remove_used_rows:
                to_keep = cp.full(data.data.shape[1], False)
                to_keep[self.remove_background_n_rows:-self.remove_background_n_rows] = True
                data.data = data.data[:,to_keep,:]
            if self.remove_background_based_on_rows:
                data.data -= background_estimate[None,None,:]
            
        if self.remove_background_based_on_pixels:
            mean_per_pixel = cp.mean(data_for_background_removal, axis=0).flatten()
            #plt.figure();plt.semilogy(cp.sort(mean_per_pixel).get())
            inds = cp.argsort(cp.mean(data.data,axis=0).flatten())
            background_estimate = cp.mean(mean_per_pixel[inds[:self.remove_background_pixels]])
            if diagnostic_plots:
                print(background_estimate)
            data.data -= background_estimate
            
        if self.restore_invalids:
            data.data[:,was_invalid] = cp.nan
            
 
@dataclass
class ApplyTimeBinning(kgs.BaseClass):
    
    time_binning = 3
    
    def __call__(self, data, planet, observation_number):
        data.data = bin_first_axis(data.data, self.time_binning)
        data.times = bin_first_axis(data.times, self.time_binning)
        data.time_intervals = bin_first_axis(data.time_intervals, self.time_binning)*self.time_binning
        
        
@dataclass        
class ApplyWavelengthBinning(kgs.BaseClass):
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        if data.is_FGS:
            data.data = cp.sum(data.data, axis=(1,2))
            data.data = cp.reshape(data.data, (-1,1))
        else:
            data.data = cp.sum(data.data, axis=1)
        data.data = cp.ascontiguousarray(data.data)
        
        # Estimate noise per pixel
        data.noise_est = cp.empty((data.data.shape[1]))
        for ii in range(data.data.shape[1]):
            data.noise_est[ii] = ariel_numerics.estimate_noise_cp(data.data[:,ii])*np.sqrt(data.time_intervals[0])
        #data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])

@dataclass    
class ApplyWavelengthBinning2(kgs.BaseClass):
    
    options: object = field(init=True, default_factory = apply_pca_modelOptions)
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        if data.is_FGS:
            data.data = cp.reshape(data.data, (-1,1024,1))
            wavelength_ids = [0]
        else:
            wavelength_ids = np.arange(1,283)

        data.data = apply_pca_model(data.data, wavelength_ids, self.options, residual_mode=0)[0]
        
        # Estimate noise per pixel        
        data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])
        
AIRS_C0 = kgs.dill_load(kgs.calibration_dir + 'AIRS_C0_2.pickle')
AIRS_C = kgs.dill_load(kgs.calibration_dir + 'AIRS_jitter.pickle')[0]
AIRS_design_matrix = cp.concatenate([AIRS_C0.reshape(1,32,282), cp.array(AIRS_C[:2,:]).reshape(2,32,282)])
AIRS_design_matrix_np = AIRS_design_matrix.get()
AIRS_rr = [cp.array(c) for c in kgs.dill_load(kgs.calibration_dir + 'AIRS_rr2.pickle')]
del AIRS_C0; del AIRS_C;
class ApplyWavelengthBinningAIRS2(kgs.BaseClass):
    residual_threshold = np.inf
    combine_rr2 = False
    cutoff_sum=0
    use_noise_est_naive = False
    sequential_fit = False
    #alpha=-0.5
    
    # Diagnostics
    #residual = None
    
    @kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        assert not data.is_FGS
        
        dataa = data.data
        
        result = cp.empty((dataa.shape[0], dataa.shape[2]))
        residual = cp.empty_like(dataa)
        residual_expected = cp.zeros((32,282))
        mean_vals = cp.zeros(282)
        noise_est_full = cp.zeros((32,282))
        noise_est_naive = cp.zeros((32,282))
        
        planet.diagnostics['AIRS_fallback'] = False
        
        for i_wavelength in range(282):
        
            isnan = cp.isnan(dataa[0,:,i_wavelength]).get()
            
            # Determine noise
            rhs = ariel_numerics.estimate_noise_cov_cp(dataa[:,:,i_wavelength]).get()
            rhs[isnan,:] = 0
            rhs[:,isnan] = 0        
            rhs=rhs.flatten()
            
            if diagnostic_plots and i_wavelength==0:
                plt.figure()
                plt.imshow(rhs.reshape(32,32))
                plt.title('Raw covariance')
                plt.colorbar()

           # plt.figure()
           # plt.imshow(rhs.reshape(32,32).get())
            
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
            if diagnostic_plots and i_wavelength==0:                
                plt.figure()
                plt.imshow(residual_cov.reshape(32,32))
                plt.title('Covariance residual')
                plt.colorbar()
            noise_est_naive[:,i_wavelength] = 0.4*cp.sqrt(64+cp.abs(cp.mean(dataa[:,:,i_wavelength],0)))
            noise_est2 = cp.array(coeffs[3:])
            to_change = noise_est2<noise_est_naive[:,i_wavelength]**2
            noise_est2[to_change] = (noise_est_naive[:,i_wavelength]**2)[to_change]
            noise_est = cp.sqrt(noise_est2)
            noise_est[isnan] = 0
            assert not cp.any(cp.isnan(noise_est))
#             coeffs[3:][isnan] = 0
#             if np.all(coeffs[3:][~isnan]>6):
#                 noise_est = np.sqrt(coeffs[3:])          
#                 noise_est = cp.array(noise_est)                
#             else:
#                 # fallback
#                 print('AIRS fallback')
#                 noise_est = noise_est_naive[:,i_wavelength]
#                 planet.diagnostics['AIRS_fallback'] = True
            
            noise_est_full[:,i_wavelength] = noise_est
            
            #noise_est_naive[:,i_wavelength][noise_est_naive[:,i_wavelength]<8] = 8
            
            residual_cov_ratio = residual_cov/cp.max(rhs)
            #kgs.sanity_check(np.min, noise_est[~isnan], 'noise_est_min', 8, [6,12])
            kgs.sanity_check(kgs.rms, residual_cov_ratio, 'residual_cov_rms', 1, [0,0.02])
            kgs.sanity_check(lambda x:np.max(np.abs(x)), residual_cov_ratio, 'residual_cov_max', 2, [0,0.2])
            
            if self.use_noise_est_naive:
                noise_est = noise_est_naive[:,i_wavelength]
            
            mean_handled = False
            isnan = cp.isnan(dataa[0,:,i_wavelength])
            while True:

                design_matrix = AIRS_design_matrix[:,:,i_wavelength]
                if not mean_handled:
                    design_matrix = cp.concatenate([design_matrix, cp.ones((1,32))])                      

                N = design_matrix.shape[0]

                if self.sequential_fit and mean_handled:
                    res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(dataa[:,:,i_wavelength].T, design_matrix[1:,:].T, return_A_pinv_w=True, sigma=noise_est)
                    coeffs = res[0]
                    assert not cp.any(cp.isnan(coeffs))
                    intermediate_residual = (dataa[:,:,i_wavelength].T - design_matrix[1:,:].T@coeffs).T    
                    res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(intermediate_residual.T, design_matrix[:1,:].T, return_A_pinv_w=True, sigma=noise_est)
                    coeffs = res[0]
                    assert not cp.any(cp.isnan(coeffs))
                    residual[:,:,i_wavelength] = (intermediate_residual.T - design_matrix[:1,:].T@coeffs).T                       
                else:
                    res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(dataa[:,:,i_wavelength].T, design_matrix.T, return_A_pinv_w=True, sigma=noise_est)
                    coeffs = res[0]
                    assert not cp.any(cp.isnan(coeffs))
                    residual[:,:,i_wavelength] = (dataa[:,:,i_wavelength].T - design_matrix.T@coeffs).T    
                
                    A_pinv_w = res[1]
                    A_pinv_w_full = cp.zeros((N,32))
                    A_pinv_w_full[:,~cp.isnan(dataa[0,:,i_wavelength])] = A_pinv_w                
                    mat = design_matrix.T@A_pinv_w_full
                    cov_expected = cp.diag(noise_est**2) - mat@cp.diag(noise_est**2)@mat.T
                    cov_expected[cov_expected<0] = 0
                    residual_expected[:,i_wavelength] = cp.sqrt(cp.diag(cov_expected))
                    assert not cp.any(cp.isnan(residual_expected[:,i_wavelength]))

                    residual_expected_ratio = cp.mean(residual[:,:,i_wavelength],0)/residual_expected[:,i_wavelength]*np.sqrt(dataa.shape[0])
                    residual_expected_ratio[cp.isnan(residual_expected_ratio)] = 0
                
                if not (self.sequential_fit and mean_handled) and cp.any(cp.abs(residual_expected_ratio)>self.residual_threshold):
                    #print(i_wavelength, cp.max(cp.abs(residual_expected_ratio)), cp.argmax(cp.abs(residual_expected_ratio)), mean_handled)
                    dataa[:,cp.argmax(cp.abs(residual_expected_ratio)),i_wavelength] = cp.nan
                else:
                    #mean_handled = True
                    if not mean_handled:
                        #print(coeffs[3,:].shape)
                        dataa[:,:,i_wavelength]-=cp.mean(coeffs[3,:])    
                        mean_vals[i_wavelength] = cp.mean(coeffs[3,:])    
                        mean_handled = True
                    else:
                        #result[:,i_wavelength] = self.alpha*np.sum(coeffs*AIRS_rr[i_wavelength][:,None],0)+(1-self.alpha)*coeffs[0,:]
                        if self.sequential_fit:
                            result[:,i_wavelength] = coeffs[0,:]
                        elif self.combine_rr2:
                            result[:,i_wavelength] = cp.sum(coeffs*AIRS_rr[i_wavelength][:,None],0)#coeffs[0,:]
                        else:
                            reconstructed_signal = (design_matrix.T@coeffs).T
                            result[:,i_wavelength] = cp.sum(reconstructed_signal[:,self.cutoff_sum:32-self.cutoff_sum],1)
                        #result[:,i_wavelength] = coeffs[0,:]
                        #result[:,i_wavelength] = np.sum(coeffs*AIRS_rr[i_wavelength][:,None],0)#coeffs[0,:]
                        break
                        # if self.use_rr:
                        #     result[:,i_wavelength] = np.sum(coeffs*AIRS_rr[i_wavelength][:,None],0)#coeffs[0,:]
                        # else:
                        #     result[:,i_wavelength] = coeffs[0,:]
                        # break
                        
        #kgs.sanity_check(np.nanmin, noise_est_full/noise_est_naive, 'noise_est_ratio', 3, [0.4,0.9])
        kgs.sanity_check(np.nanmax, 1-cp.std(residual,0)/residual_expected, 'residual_std_ratio', 4, [0.03,0.15])
        kgs.sanity_check(lambda x:np.nanmax(np.abs(x)), cp.mean(residual,0)/residual_expected, 'residual_mean_ratio', 5, [0,5])
              
        if diagnostic_plots:
            plt.figure()
            plt.scatter(noise_est_naive.flatten().get(), noise_est_full.flatten().get())
            plt.axline((0,0), slope=1, color='black')
            plt.figure()
            plt.imshow(residual_expected.get(), interpolation='none', aspect='auto')
            plt.colorbar()
            plt.title('Residual expected')
            plt.figure()
            plt.imshow((cp.std(residual,0)/residual_expected).get(), interpolation='none', aspect='auto')
            plt.colorbar()
            plt.title('Residual STD scaled')
            plt.figure()
            #plt.imshow((cp.mean(residual,0)/residual_expected*np.sqrt(dataa.shape[0]))[:,:].get(), interpolation='none', aspect='auto')
            plt.imshow((cp.mean(residual,0)/residual_expected)[:,:].get(), interpolation='none', aspect='auto')
            #plt.clim([-30,30])
            plt.colorbar()          
            plt.title('Residual mean scaled')
            # plt.figure()
            # plt.plot(mean_vals.get())
            plt.figure()
            plt.imshow(cp.log(noise_est_full).get(), interpolation='none', aspect='auto')
            plt.colorbar()
            print('min', cp.min(noise_est_full), cp.nanmin(noise_est_full))
            plt.title('Noise est')
            for ii in range(1):
                plt.figure()
                for jj in range(282):
                    i_wavelength = 10*ii+jj
                    plt.scatter(noise_est_full[:,i_wavelength].flatten().get(), noise_est_naive[:,i_wavelength].flatten().get())
                plt.axline((0,0), slope=1, color='black')
        data.data = result
        data.noise_est = ariel_numerics.estimate_noise_cp(data.data)*np.sqrt(data.time_intervals[0])
        
       # self.residual = residual


    

def default_loaders():
    loader = kgs.TransitLoader()
    loader.load_raw_data = LoadRawData()
    loader.apply_pixel_corrections = ApplyPixelCorrections()
    loader.apply_full_sensor_corrections = ApplyFullSensorCorrections()
    loader.apply_time_binning = ApplyTimeBinning()
    loader.apply_wavelength_binning = ApplyWavelengthBinning()
    
    loaders = [loader, copy.deepcopy(loader)]
    
    # FGS configuration   
    loaders[0].apply_full_sensor_corrections.inpainting_2d = True
    loaders[0].apply_full_sensor_corrections.restore_invalids = True
    loaders[0].apply_full_sensor_corrections.remove_background_based_on_pixels = True    
    loaders[0].apply_time_binning.time_binning = 50
    loaders[0].apply_wavelength_binning = ariel_load_FGS.ApplyWavelengthBinningFGS2()

    # AIRS configuration
    loaders[1].apply_pixel_corrections.clip_columns=True
    loaders[1].apply_pixel_corrections.clip_columns1=39
    loaders[1].apply_pixel_corrections.clip_columns2=321    
    loaders[1].apply_time_binning.time_binning = 5
    loaders[1].apply_wavelength_binning = ApplyWavelengthBinningAIRS2()
     
    return loaders


def raw_data_diagnostics(data, observation_number, loaders):
    
    global diagnostic_plots
    diagnostic_plots = True
    ariel_load_FGS.diagnostic_plots = True
    
    transit = data.transits[observation_number]
    
    transit.load_to_step(0, data, loaders)
    transit.load_to_step(2, data, loaders)
    
    plt.figure()
    plt.imshow(cp.log(cp.mean(transit.data[0].data,0)).get())
    plt.colorbar()
    plt.title('FGS mean over time')
    
    plt.figure()
    plt.imshow(cp.log(cp.mean(transit.data[1].data,0)).get(), aspect='auto', interpolation='none')
    plt.colorbar()
    plt.title('AIRS mean over time')
    
     
    def plots_on_full_signal():
        _,ax = plt.subplots(1,3,figsize=(24,8))
        plt.sca(ax[0])
        plt.plot(cp.nanmean(transit.data[0].data,(1,2)).get())
        plt.sca(ax[1])
        M=cp.mean(transit.data[1].data,1)
        plt.imshow(M.get().T, aspect='auto', interpolation='none')
        plt.sca(ax[2])
        plt.imshow((M-np.mean(M,0)).get().T, aspect='auto', interpolation='none')
    plots_on_full_signal()
    
    transit.load_to_step(3, data, loaders)
    plots_on_full_signal()
    
    transit.load_to_step(4, data, loaders)
    plots_on_full_signal()
    
    transit.load_to_step(5, data, loaders)
    
    transit.load_to_step(0, data, loaders)
    
    diagnostic_plots = False
    ariel_load_FGS.diagnostic_plots = False
    
   
        
        