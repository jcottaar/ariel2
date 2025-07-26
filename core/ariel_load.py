import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import copy
from dataclasses import dataclass, field, fields
import kaggle_support as kgs
import pyarrow.parquet
from astropy.stats import sigma_clip

def read_parquet_to_numpy(filename):
    table = pyarrow.parquet.read_table(filename)
    return table.to_pandas().to_numpy()

@dataclass
class LoadRawData(kgs.BaseClass):   
    
    def __call__(self, data, planet, observation_number):
        filename = planet.get_directory() + kgs.sensor_names[not data.is_FGS] + '_signal_' + str(observation_number) + '.parquet'
        data.data = read_parquet_to_numpy(filename)
        if data.is_FGS:
            data.data = np.reshape(data.data, (135000, 32, 32))
        else:
            data.data = np.reshape(data.data, (11250, 32, 356))                  
            
        data.times = kgs.axis_info[kgs.sensor_names[not data.is_FGS]+'-axis0-h'].to_numpy()*3600
        data.times = data.times[~np.isnan(data.times)]
        
        if not data.is_FGS:
            dt = kgs.axis_info['AIRS-CH0-integration_time'].dropna().values
        else:
            dt = np.ones(data.data.shape[0])*0.1
        dt[1::2] += 0.1
        data.time_intervals = dt
        
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
    
    adc_offset_sign = -1 
    dark_current_sign = 1
    
    @kgs.profile_each_line
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
            kgs.sanity_check(lambda x:x, gain, 'gain', 1, [0.3, 3])                
            kgs.sanity_check(lambda x:x, offset, 'offset', 1, [-3000, 3000])
            return signal

        def mask_hot_dead(signal, dead, dark):
            hot = sigma_clip(dark, sigma=self.hot_sigma_clip, maxiters=5).mask
            if kgs.sanity_checks_active:
                kgs.sanity_check(lambda x:x, np.mean(hot), 'ratio_hot', 2, [0, 0.01])        
                kgs.sanity_check(lambda x:x, np.mean(dead), 'ratio_dead', 2, [0, 0.01])      
            if self.mask_hot:
                signal[:, hot] = np.nan
            if self.mask_dead:
                signal[:, dead] = np.nan
            return signal

        def apply_linear_corr(lc,x):        
            result =  lc[0,:,:]+x*(lc[1,:,:]+x*(lc[2,:,:]+x*(lc[3,:,:]+x*(lc[4,:,:]+x*lc[5,:,:]))))
            if kgs.sanity_checks_active:
                kgs.sanity_check(np.nanmax, np.abs(result-x), 'linear_corr_impact', 1, [0, 50000])
            return result

        def clean_dark(signal, dark, dt):    
            signal -= self.dark_current_sign * dark * dt[:, np.newaxis, np.newaxis]
            kgs.sanity_check(np.min, gain, 'dark_min', 1, [0.3, 3]) 
            kgs.sanity_check(np.max, gain, 'dark_max', 1, [0.3, 3])        
            return signal

        def correct_flat_field(flat, signal):        
            flat = np.tile(flat, (signal.shape[0], 1, 1))
            signal = signal / flat
            kgs.sanity_check(np.min, flat, 'flat_min', 1, [-1, 1.5]) 
            kgs.sanity_check(np.max, flat, 'flat_max', 1, [0.7, 1.5])                
            return signal
        
        # Load calibration data
        dark = clip_2d(read_parquet_to_numpy(calibration_directory+'dark.parquet').astype(np.float64))
        dead = clip_2d(read_parquet_to_numpy(calibration_directory+'dead.parquet').astype(np.float64))
        flat = clip_2d(read_parquet_to_numpy(calibration_directory+'flat.parquet').astype(np.float64))
        linear_corr = clip_3d(read_parquet_to_numpy(calibration_directory+'linear_corr.parquet').astype(np.float64).reshape(6,32,-1))
        
        # Clip to desired rows and columns, and make float64
        data.data = clip_3d(data.data).astype(np.float64)
        
        # ADC
        offset = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_offset'].to_numpy()
        gain = kgs.adc_info[kgs.sensor_names[not data.is_FGS]+'_adc_gain'].to_numpy()
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
        print(data.time_intervals, data.times)
      
        
        
            
        
def default_loaders():
    loader = kgs.TransitLoader()
    loader.load_raw_data = LoadRawData()
    loader.apply_pixel_corrections = ApplyPixelCorrections()
    
    loaders = [loader, copy.deepcopy(loader)]
    
    # AIRS configuration
    loaders[1].apply_pixel_corrections.clip_columns=True
    loaders[1].apply_pixel_corrections.clip_columns1=39
    loaders[1].apply_pixel_corrections.clip_columns2=321    
    
    return loaders