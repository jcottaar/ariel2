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

pca_data = kgs.dill_load(kgs.temp_dir + '/explore_bad_pixels_pca.pickle')[1]

@kgs.profile_each_line
def remove_bad_pixels_pca(data, components, n_components, noise_est_threshold, residual_threshold, also_remove_mean):
    # data: (time) x (pixel) x (wavelength)
    # modifies in place
    for i_wavelength in range(data.shape[2]):
        this_data = data[:,:,i_wavelength]        
        n_removed=0        
        noise_est = cp.ones(this_data.shape[1])
        #for ii in range(this_data.shape[1]):
        #   noise_est[ii] = ariel_numerics.estimate_noise_cp(this_data[:,ii])
        NN=100
        for ii in range(this_data.shape[1]//NN+1):            
            maxind=min((this_data.shape[1], (ii+1)*NN))
            noise_est[ii*NN:maxind] = ariel_numerics.estimate_noise_cp(this_data[:,ii*NN:maxind])
       # noise_est[cp.isnan(noise_est)]=cp.nanmean(noise_est)
        noise_est[noise_est<10] = 10
        data_mean = cp.mean(this_data,0)
        noise_est_threshold_cp = cp.array(noise_est_threshold)
        n_nan = cp.sum(cp.isnan(this_data))
        this_data[:,(data_mean<0) | (noise_est>noise_est_threshold*cp.sqrt(data_mean))] = cp.nan
        if cp.sum(cp.isnan(this_data))!=n_nan:
            print('removing noise_est')
        # for ii in range(this_data.shape[1]):
        #    if (data_mean[ii]<0 or noise_est[ii]>noise_est_threshold*cp.sqrt(data_mean[ii])):
        # #       #print(i_wavelength, 'noise_est', noise_est[ii]/cp.sqrt(data_mean[ii]))
        #        this_data[:,ii] = cp.nan
        #        #noise_est[ii] = cp.nan
        while True:            
            #this_data[:,15*32+15]=0
            #print(this_data.shape)
            data_mean = cp.mean(this_data,0)
            #print(data_mean.shape)
            design_matrix = cp.stack([cp.ones(data_mean.shape[0])]+[c[0,:] for c in components[i_wavelength][0:n_components]]).T#[
            #design_matrix[:,1:] = design_matrix[:,1:]-cp.mean(design_matrix[:,1:],1)
            #print(design_matrix.shape)            
            coeffs = ariel_numerics.lstsq_nanrows_normal_eq_with_pinv_sigma(data_mean[:,None], design_matrix, sigma=noise_est, return_A_pinv_w=False)
            residual = (data_mean[:,None]-design_matrix@coeffs[0])[:,0]
            residual_scaled = residual/noise_est*cp.sqrt(this_data.shape[0])
            residual_scaled[cp.isnan(residual_scaled)] = 0
            if cp.any(cp.abs(residual_scaled)>residual_threshold):
                print('removing residual')
                #print(i_wavelength, cp.max(cp.abs(residual_scaled)))
                this_data[:,cp.argmax(cp.abs(residual_scaled))] = cp.nan
                #noise_est[cp.argmax(cp.abs(residual_scaled))] = cp.nan
                n_removed+=1
            else:
                break
        if also_remove_mean:
            #print('coeffs0', coeffs[0][0][0])
            this_data[...] -= coeffs[0][0][0]

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
    mask_hot = True
    hot_sigma_clip = 5
    linear_correction = True
    dark_current = True
    flat_field = True
    remove_cosmic_rays = True
    cosmic_ray_threshold = 20
    remove_first_and_last_frame = False
    
    adc_offset_sign = 1 
    dark_current_sign = 1
    
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

@dataclass
class ApplyFullSensorCorrections(kgs.BaseClass):
    
    remove_bad_pixels_pca = False
    remove_bad_pixels_pca_inputs = None
    
    inpainting_time = True
    inpainting_wavelength = False
    inpainting_2d = False
    
    #remove_constant = 0.
    
    remove_background_based_on_rows = False
    remove_background_n_rows = 8 
    remove_background_remove_used_rows = True
    
    remove_background_based_on_pixels = False
    remove_background_pixels = 100
    
    #@kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        if self.remove_bad_pixels_pca:
            if data.is_FGS:                
                dat = cp.reshape(data.data, (-1,1024))[:,:,None]
                remove_bad_pixels_pca(dat, *self.remove_bad_pixels_pca_inputs)
            else:
                remove_bad_pixels_pca(data.data, *self.remove_bad_pixels_pca_inputs)
        assert self.inpainting_time # actually done above
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
        if not data.is_FGS:
            background_data = cp.concatenate((data.data[:,:self.remove_background_n_rows,:], data.data[:,-self.remove_background_n_rows:,:]), axis=1)
            background_estimate = cp.mean(background_data, axis=(0,1))
            if self.remove_background_remove_used_rows:
                to_keep = cp.full(data.data.shape[1], False)
                to_keep[self.remove_background_n_rows:-self.remove_background_n_rows] = True
                data.data = data.data[:,to_keep,:]
            if self.remove_background_based_on_rows:
                data.data -= background_estimate[None,None,:]
            
        if self.remove_background_based_on_pixels:
            mean_per_pixel = cp.mean(data.data, axis=0).flatten()
            #plt.figure();plt.semilogy(cp.sort(mean_per_pixel).get())
            background_estimate = cp.mean(cp.sort(mean_per_pixel)[:self.remove_background_pixels])
            #print(background_estimate)
            data.data -= background_estimate
 
@dataclass
class ApplyTimeBinning(kgs.BaseClass):
    
    time_binning = 3
    
    def __call__(self, data, planet, observation_number):
        data.data = bin_first_axis(data.data, self.time_binning)
        data.times = bin_first_axis(data.times, self.time_binning)
        data.time_intervals = bin_first_axis(data.time_intervals, self.time_binning)*self.time_binning
        
        
        
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
            
        
def default_loaders():
    loader = kgs.TransitLoader()
    loader.load_raw_data = LoadRawData()
    loader.apply_pixel_corrections = ApplyPixelCorrections()
    loader.apply_full_sensor_corrections = ApplyFullSensorCorrections()
    loader.apply_time_binning = ApplyTimeBinning()
    loader.apply_wavelength_binning = ApplyWavelengthBinning()
    
    loaders = [loader, copy.deepcopy(loader)]
    
    # FGS configuration   
    loaders[0].apply_full_sensor_corrections.inpainting_2d=True
    loaders[0].apply_full_sensor_corrections.remove_background_based_on_pixels = False
    loaders[0].apply_time_binning.time_binning = 50
    loaders[0].apply_full_sensor_corrections.remove_bad_pixels_pca_inputs = [pca_data[0:1],4,100,5,False]
    
    # AIRS configuration
    loaders[1].apply_pixel_corrections.clip_columns=True
    loaders[1].apply_pixel_corrections.clip_columns1=39
    loaders[1].apply_pixel_corrections.clip_columns2=321    
    loaders[1].apply_full_sensor_corrections.inpainting_wavelength=True
    loaders[1].apply_full_sensor_corrections.remove_background_based_on_rows=False
    loaders[1].apply_time_binning.time_binning = 5
    loaders[1].apply_full_sensor_corrections.remove_bad_pixels_pca_inputs = [pca_data[1:],3,100,5,False]
    
    return loaders


def raw_data_diagnostics(data, observation_number, loaders):
    
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
    
    transit.load_to_step(0, data, loaders)
    
   
        
        