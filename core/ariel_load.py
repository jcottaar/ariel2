'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module implements data loading, from raw parquet files to wavelength-binned data.
'''

import numpy as np
import cupy as cp
import copy
from dataclasses import dataclass, field
import kaggle_support as kgs
import pyarrow.parquet
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import ariel_numerics
import ariel_load_FGS
import ariel_load_AIRS

def default_loaders():
    # Define the default loaders, based on the classes below.
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
    loaders[1].apply_full_sensor_corrections.pca_options.n_components = 2
    loaders[1].apply_full_sensor_corrections.remove_background_based_on_rows = True
    loaders[1].apply_full_sensor_corrections.inpainting_wavelength = True
    loaders[1].apply_full_sensor_corrections.restore_invalids = True
    loaders[1].apply_wavelength_binning = ariel_load_AIRS.ApplyWavelengthBinningAIRS3()
     
    return loaders

@dataclass
class LoadRawData(kgs.BaseClass):  
    # Loads raw parquet files
    
    def __call__(self, data, planet, observation_number):
        filename = planet.get_directory() + kgs.sensor_names[not data.is_FGS] + '_signal_' + str(observation_number) + '.parquet'
        data.data = read_parquet_to_cupy(filename)
        if data.is_FGS:
            data.data = cp.reshape(data.data, (135000, 32, 32))
        else:
            data.data = cp.reshape(data.data, (11250, 32, 356))                  

        # Times
        data.times = cp.array(kgs.axis_info[kgs.sensor_names[not data.is_FGS]+'-axis0-h'].to_numpy())*3600
        data.times = data.times[~cp.isnan(data.times)]
        
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

@dataclass
class ApplyPixelCorrections(kgs.BaseClass):
    # Applies per-pixel corrections
    
    clip_columns = False # Clip columns of signal?
    clip_columns1 = None
    clip_columns2 = None
    
    clip_rows = False # Clip rows of signal?
    clip_rows1 = None
    clip_rows2 = None
    
    mask_dead = True # Remove dead pixels?
    mask_hot = False # Remove hot pixels?
    hot_sigma_clip = 5
    linear_correction = True # Apply linear correction?
    dark_current = True # Apply dark current correction?
    flat_field = True # Apply flat field correction?
    remove_cosmic_rays = True # Remove cosmic rays?
    cosmic_ray_threshold = 20 # sigma threshold for removing cosmic rays?
    remove_last_frame = True # Remove last frame from data? It is wrong if part of the transit
    
    adc_offset_sign = 1 # Change ADC offset sign?
    dark_current_sign = 1 # Change dark current sign?

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
            # Remove comsic rays based on sigma filter
            cosmic_ray_count = 0
            for ii in range(signal.shape[1]):
                signal_noise = ariel_numerics.remove_trend_cp(signal[:,ii,...]) # take only high frequent content
                is_cosmic_ray = cp.abs(signal_noise - cp.mean(signal_noise,0))/cp.std(signal_noise,0) > self.cosmic_ray_threshold
                signal[:,ii,...][is_cosmic_ray] = cp.nan
                cosmic_ray_count += cp.sum(is_cosmic_ray)                  
            kgs.sanity_check(lambda x:cp.max(cp.mean(x)), cosmic_ray_count/np.prod(signal.shape), 'cosmic_ray_removal', 10, [0,5e-6])            
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
        
        # ADC
        offset = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_offset'].to_numpy()[0]
        gain = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_gain'].to_numpy()[0]
        data.data = ADC_convert(data.data,gain,offset)
        
        # Mask
        data.data = mask_hot_dead(data.data,dead>0,dark)
        
        # Linear correction
        if self.linear_correction:
            data.data = apply_linear_corr(linear_corr, data.data)
            
        # Dark current
        if self.dark_current:            
            data.data = clean_dark(data.data,dark,data.time_intervals)            
            
        # Flat field
        if self.flat_field:
            data.data = correct_flat_field(flat, data.data)
                 
        # Correlated double sampling    
        data.data = data.data[1::2,:,:]-data.data[0::2,:,:]
        data.times = data.times[0::2]/2+data.times[1::2]/2   
        data.time_intervals = data.time_intervals[1::2]
        
        # Remove cosmic rays
        if self.remove_cosmic_rays:
            data.data = remove_cosmic_rays(data.data)
            inpaint_along_axis_inplace(data.data,0) # Fill cosmic ray datapoints with interpolation
        
        # Flip AIRS to have ascending wavelengths
        if not data.is_FGS:
            data.data = cp.flip(data.data, axis=2)
            data.data = cp.ascontiguousarray(data.data)
            data.wavelengths = cp.flip(data.wavelengths)
            
        # Remove last frame
        if self.remove_last_frame:
            data.data = data.data[:-1,...]
            data.times = data.times[:-1]
            data.time_intervals = data.time_intervals[:-1]            
        

@dataclass
class ApplyFullSensorCorrections(kgs.BaseClass):
    # Apply corrections that involve multiple pixels
        
    inpainting_wavelength = False # Apply 1D inpainting along wavelength axis
    inpainting_2d = False # Apply 2D inpainting
    restore_invalids = False # Make inpainted points invalid again (after applying background correction)
    
    # Settings for foreground correction
    use_pca_for_background_removal = True
    pca_options: object = field(init=True, default_factory = lambda:apply_pca_modelOptions(n_components=4))    
    remove_background_based_on_rows = False # Remove background based on the top and bottom rows of the AIRS sensor
    remove_background_n_rows = 8 
    remove_background_remove_used_rows = False # Remove the background rows from the signal
    
    remove_background_based_on_pixels = False # Remove background based on the darkest pixels of the FGS sensor
    remove_background_pixels = 100
    
    def __call__(self, data, planet, observation_number):        
        # Inpainting
        was_invalid = cp.isnan(data.data[0,...]) # to restore later
        if self.inpainting_wavelength:
            inpaint_vectorized(data.data)
        if self.inpainting_2d:     
            temp = cp.transpose(copy.deepcopy(data.data), (0,2,1))
            inpaint_vectorized(data.data)
            inpaint_vectorized(temp)
            data.data = (data.data+cp.transpose(temp, (0,2,1)))/2
        
        # Background correction, including attempt to correct jitter. Messy and not well documented, see writeup for (limited) description.
        if self.use_pca_for_background_removal:
            if data.is_FGS:
                data_pca = cp.reshape(data.data, (-1,1024,1))
                wavelength_ids = [0]
            else:
                data_pca = data.data
                wavelength_ids = np.arange(1,283)
            data_for_background_removal = apply_pca_model(data_pca, wavelength_ids, self.pca_options, residual_mode=2)[1]  
            data_for_background_removal = cp.reshape(data_for_background_removal, data.data.shape)                
        else:
            data_for_background_removal = data.data                
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
            inds = cp.argsort(cp.mean(data.data,axis=0).flatten())
            background_estimate = cp.mean(mean_per_pixel[inds[:self.remove_background_pixels]])
            data.data -= background_estimate
            
        # Restore invalids
        if self.restore_invalids:
            data.data[:,was_invalid] = cp.nan
            
 
@dataclass
class ApplyTimeBinning(kgs.BaseClass):
    # Apply time binning
    
    add_last_frame = True # Include the last frame if the binning doesn't work out neatly?
    time_binning = 3 # Size of bins
    
    def __call__(self, data, planet, observation_number):
        data_new = bin_first_axis(data.data, self.time_binning)
        times_new = bin_first_axis(data.times, self.time_binning)
        time_intervals_new = bin_first_axis(data.time_intervals, self.time_binning)*self.time_binning
        # Add the last frame
        if self.add_last_frame:
            ind_done = data_new.shape[0]*self.time_binning
            if ind_done < data.data.shape[0]:
                data_new = cp.concatenate((data_new, cp.mean(data.data[ind_done:data.data.shape[0],...],0)[None,...]))
                times_new = cp.concatenate((times_new, cp.mean(data.times[ind_done:data.data.shape[0]])[None]))
                time_intervals_new =  cp.concatenate((time_intervals_new, (data.time_intervals[0]*(data.data.shape[0]-ind_done))[None]))
        data.data = data_new
        data.times = times_new
        data.time_intervals = time_intervals_new
                
        
        
@dataclass        
class ApplyWavelengthBinning(kgs.BaseClass):
    # Apply simple wavelength binning. More advanced binning including jitter correction is used in submission (see below).
    
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


'''
Support functions
'''
def read_parquet_to_numpy(filename):
    # Faster way to read parquet
    table = pyarrow.parquet.read_table(filename)
    data = table.to_pandas().to_numpy()
    return data
def read_parquet_to_cupy(filename):
    return cp.array(read_parquet_to_numpy(filename))
   
def inpaint_along_axis_inplace(arr, axis=0):
    # Apply inpainting along a given axis (linear interpolation)
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
    

def inpaint_vectorized(data):
    """
    Inpaints NaN‐patches along the last axis (Y) of `data` (shape C×X×Y)
    by:
      • flat‐extrapolating any NaN runs at the left or right edge
      • linearly interpolating any interior NaN runs
    Loops only over X and over NaN‐segments (rare), otherwise fully GPU‐vectorized.
    Basically vectorized version of inpaint_along_axis_inplace.
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
     
'''
The code below, as well as the code in ariel_load_AIRS.py and ariel_load_FGS.py, deals with jitter correction.
Unfortunately, I wasn't able to get this working properly. What I have here is better than naive binning, but not by much.
I am sure it is possible to do much better.
See my writeup for some indication of what I was going for.
'''
        
pca_data = kgs.dill_load(kgs.calibration_dir + '/explore_bad_pixels_pca.pickle')
coeff_data = kgs.dill_load(kgs.calibration_dir+'/explore_base_shape_from_pca_coeff_list.pickle')
core_shapes = []
for i_wavelength in range(283):
    core_shapes.append((cp.stack([c[0,:] for c in pca_data[1][i_wavelength]]).T @ coeff_data[i_wavelength][2][:,None])[:,0])
    
@dataclass
class apply_pca_modelOptions(kgs.BaseClass):
    n_components: int = field(init=True, default=0)
    use_sum: bool = field(init=True, default=False)    

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
        res = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(this_data.T, design_matrix, sigma=noise_est, return_A_pinv_w=False)
        coeffs = res[0].T      
        if not options.use_sum:
            weighted_coeffs[:,i_data] = coeffs @ coeff_data[i_wavelength][1][options.n_components-1]
        else:
            weighted_coeffs[:,i_data] = cp.sum((design_matrix@coeffs.T).T,1)
        if residual_mode == 1:
            residual[:,:,i_data] = (this_data.T-design_matrix@coeffs.T).T
        elif residual_mode == 2:
            residual[:,:,i_data] = this_data - weighted_coeffs[:,i_data][:,None] * core_shapes[i_wavelength]
        
    output = (weighted_coeffs, residual)
    return output
