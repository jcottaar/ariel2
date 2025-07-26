import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import copy
from dataclasses import dataclass, field, fields
import kaggle_support as kgs
import pyarrow.parquet
from astropy.stats import sigma_clip

#@kgs.profile_each_line
def read_parquet_to_numpy(filename):
    table = pyarrow.parquet.read_table(filename)
    data = table.to_pandas().to_numpy()
    return data

def read_parquet_to_cupy(filename):
    return cp.array(read_parquet_to_numpy(filename))

@dataclass
class LoadRawData(kgs.BaseClass):   
    
    @kgs.profile_each_line
    def __call__(self, data, planet, observation_number):
        filename = planet.get_directory() + kgs.sensor_names[not data.is_FGS] + '_signal_' + str(observation_number) + '.parquet'
        data.data = read_parquet_to_cupy(filename)
        if kgs.profiling: cp.cuda.Device().synchronize()
        if data.is_FGS:
            data.data = cp.reshape(data.data, (135000, 32, 32))
        else:
            data.data = cp.reshape(data.data, (11250, 32, 356))                  
        if kgs.profiling: cp.cuda.Device().synchronize()
            
        data.times = cp.array(kgs.axis_info[kgs.sensor_names[not data.is_FGS]+'-axis0-h'].to_numpy())*3600
        if kgs.profiling: cp.cuda.Device().synchronize()
        data.times = data.times[~cp.isnan(data.times)]
        if kgs.profiling: cp.cuda.Device().synchronize()
        
        if not data.is_FGS:
            dt = cp.array(kgs.axis_info['AIRS-CH0-integration_time'].dropna().values)
        else:
            dt = cp.ones(data.data.shape[0])*0.1
        dt[1::2] += 0.1
        data.time_intervals = dt
        if kgs.profiling: cp.cuda.Device().synchronize()
        
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
            kgs.sanity_check(lambda x:x, gain, 'gain', 1, [0.4, 0.5])                
            kgs.sanity_check(lambda x:x, offset, 'offset', 2, [-1001, -999])
            return signal

        def mask_hot_dead(signal, dead, dark):
            hot = cp.array(sigma_clip(dark.get(), sigma=self.hot_sigma_clip, maxiters=5).mask)
            if kgs.sanity_checks_active:
                kgs.sanity_check(lambda x:x, cp.mean(hot), 'ratio_hot', 3, [0, 0.015])        
                kgs.sanity_check(lambda x:x, cp.mean(dead), 'ratio_dead', 4, [0, 0.005])      
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
            kgs.sanity_check(cp.min, dark[~cp.isnan(signal[0,:,:])], 'dark_min', 5, [-0.0015, 0.004]) 
            kgs.sanity_check(cp.max, dark[~cp.isnan(signal[0,:,:])], 'dark_max', 6, [0., 0.02])        
            return signal

        def correct_flat_field(flat, signal):        
            signal = signal / flat[cp.newaxis, :,:]
            kgs.sanity_check(cp.min, flat[~cp.isnan(signal[0,:,:])], 'flat_min', 7, [0.7, 1.1]) 
            kgs.sanity_check(cp.max, flat[~cp.isnan(signal[0,:,:])], 'flat_max', 8, [0.9, 1.2])                
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
        if kgs.profiling: cp.cuda.Device().synchronize()
        
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
        
        # One nan is all nan
        data.data[:, cp.any(cp.isnan(data.data),0)] = cp.nan
        if kgs.profiling: cp.cuda.Device().synchronize()
          
        
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