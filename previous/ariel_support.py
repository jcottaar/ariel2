'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module provides general support functionality for my ARIEL competition work, as well as data loading functions.
Sections:
- Helper functions and classes
- Globals
- Data loading and preprocessing
- Modeling framework
- Sanity checks
- Scoring functions
'''

import pandas as pd
import numpy as np
import dill as pickle
from astropy.stats import sigma_clip
import itertools
import os
import pandas.api.types
import scipy.stats
import scipy as sp
import scipy.ndimage
import multiprocess as multiprocessing
import copy
import functools
import hashlib
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


'''
Helper functions and classes
'''

# Are we on Kaggle?
if os.path.isdir('d:/ariel/data/'):
    running_on_kaggle = 0;
else:
    running_on_kaggle = 1;

# Where are our files?
def data_dir():
    if running_on_kaggle == 1:
        return '/kaggle/input/ariel-data-challenge-2024/'
    else:
        return 'd:/ariel/data/'
def file_loc():
    if running_on_kaggle==1:
        return '/kaggle/input/my-ariel-library/'
    else:
        return 'D:/ariel/code/'

# How many workers is optimal for parallel pool?
def recommend_n_workers():
    if running_on_kaggle == 1:
        return 2
    else:
        return 7

# Helper class - doesn't allow new properties after construction. Written by ChatGPT.
@dataclass
class LockedClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Mark the object as frozen after initialization
        object.__setattr__(self, '_is_frozen', True)

    def __setattr__(self, key, value):
        # If the object is frozen, prevent setting new attributes
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)

# Small wrapper for pickle loading
def pickle_load(filename):
    filehandler = open(filename, 'rb');
    data = pickle.load(filehandler)
    filehandler.close()
    return data

# Small wrapper for pickle saving
def pickle_save(filename, data):
    filehandler = open(filename, 'wb');
    data = pickle.dump(data, filehandler)
    filehandler.close()
    return data

# Moving averages
def moving_average_3d(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:,:,:] = ret[n:,:,:] - ret[:-n,:,:]
    return ret[n - 1:] / n
def moving_average_2d(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return ret[n - 1:] / n
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def gaussian_2D_filter_with_nans(U, sigma):
    # Not used in model, but useful to have around   
    # Source: https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    truncate=2.0   # truncate filter at this many sigmas
    V=U.copy()
    V[np.isnan(U)]=0
    VV=sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=sp.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)
    Z=VV/WW
    assert not np.any(np.isnan(Z))
    return Z

def rms(x):
    return np.sqrt(np.mean(x**2))


'''
Globals
'''
sanity_checks_active = True # Whether to throw errors if we see weird things in the data
sanity_checks_without_errors = False # Perform the sanity checks above but don't throw errors
sanity_checks = dict() # Global object to keep track of all sanity checks. 
never_model_parallel = False # If True, no parallel pool is used during modeling; does not affect loader

# Some input data that we only have to load once
axis_info = pd.read_parquet(data_dir()+'axis_info.parquet')
train_planet_list = pd.read_csv(data_dir() + 'train' + '_adc_info.csv')["planet_id"].to_numpy()
test_planet_list = pd.read_csv(data_dir() + 'test' + '_adc_info.csv')["planet_id"].to_numpy()


'''
Data loading and preprocessing
'''
# Define the loader to use in submission
def baseline_loader(include_later_optimization=False):
    # include_later_optizimation=False: as my best submission before the competition closed
    # include_later_optizimation=True: include later learnings
    options = LoaderOptions()

    if include_later_optimization:
        options.row_bottom = 0
        options.row_top = 32
        options.options_AIRS.correct_background = True

    return options

# General loader options applying to both AIRS and FGS
class LoaderOptions(LockedClass):

    # Which rows to use for AIRS. This belongs in options_AIRS but I'm too lazy to refactor this...
    row_bottom = 8 
    row_top = 24

    # Options per signal
    options_AIRS = 0
    options_FGS = 0

    load_FGS = True # Whether to load FGS at all

    def __post_init__(self):
        # Set defaults
        self.options_AIRS = LoaderOptionsBaseSignal()
        self.options_FGS = LoaderOptionsBaseSignal()
        self.options_FGS.signal_name = 'FGS1'

        # Detector size for FGS
        self.options_FGS.n_rows = 32*32
        self.options_FGS.n_cols = 1

        # Time binning for FGS to get similar number of timestamps as for AIRS. The fact that it doesn't match exactly is not an issue for the GP model.
        self.options_FGS.time_binning = int(np.round(self.options_AIRS.time_binning * 4.7/0.4))
        self.options_FGS.time_binning_near_transitions = int(np.round(self.options_AIRS.time_binning_near_transitions * 4.7/0.4))

        # Misc
        self.options_FGS.inpainting = False
        self.options_FGS.fix_mean_per_frame_in_column_binning = False
        
        super().__post_init__()

# Loader options applying to one sensor
class LoaderOptionsBaseSignal(LockedClass):
    signal_name = 'AIRS-CH0'

    # Detector size
    n_rows = 32
    n_cols = 356

    # Enable preprocessing steps
    mask_dead = True
    mask_hot = True
    linear_correction = True
    dark_current = True
    flat_field = True
    
    adc_offset_sign = -1 # -1 is opposite to example notebook
    dark_current_sign = 1 # 1 is as in example notebook
    n_pixels = 100 # use only the brightest pixels - only applies to FGS
    correct_background = False # correct for fixed background signal, using the top 8 rows or bottom 8 rows (AIRS) or the fixed value below (FGS)
    correct_background_fixed_fgs_value = 1. # see above
    # Note: if correct_background = True, row_bottom=0 and row_top=32 should be used in LoaderOptions
    
    correct_jitter = False # use the 'correct_jitter' function to deal with signal shifting between rows. AIRS only
    inpainting = True # fill in invalid values. AIRS only

    column_binning = -1 # how to bin per wavelength, applies to both AIRS and FGS
    # -1: simple sum
    # 0: don't bin
    # 1: weighted sum, giving less weight to noisier pixels. Only works decently with correct_jitter=True
    fix_mean_per_frame_in_column_binning = True # If true, the weighted sum above may not change the mean over the whole detector. Recommended ot use.
    
    width_for_noise_estimate_ma = 101 # moving average filter size for estimating noise per pixel

    time_binning = 21 # how many frames to bin during time binning
    time_binning_near_transitions = 5 # time_binning value to use if close to ingress/egress ('close' as defined by transition_margin below)
    transition_margin = 0.3 # see above

# Helper class to manage data loading
class DataLoader(LockedClass):
    loader_options = [] # LoaderOptions, set in constructor
    planet_ids_to_load = [0] # which planet IDs to load, set in constructor
    load_train = True # whether to load training data (rather than test)
    include_labels = True # whether to include the ground truth
    use_cache = True # whether to use cached data
    run_in_parallel = True # whether to use parallel pool for loading

    def __post_init__(self):
        self.planet_ids_to_load = train_planet_list
        self.loader_options = LoaderOptions()
        super().__post_init__()

    def load(self):
        run_on_train = 0
        if self.load_train:
            run_on_train = 1

        # Load main data, in parallel or not
        if len(self.planet_ids_to_load)>1 and self.run_in_parallel:
            p = multiprocessing.Pool(recommend_n_workers())
            data = p.starmap(load_and_preprocess, 
                zip(self.planet_ids_to_load, itertools.repeat(run_on_train), itertools.repeat(self.use_cache), itertools.repeat(self.loader_options)))  
            p.close()
        else:
            data = [load_and_preprocess(p, run_on_train,self.use_cache,self.loader_options) for p in self.planet_ids_to_load]
            
        # Load auxiliary data
        if self.load_train:
            train_str = 'train';
        else:
            train_str = 'test';
        adc_info = pd.read_csv(data_dir() + train_str + '_adc_info.csv')
        adc_info = adc_info.set_index('planet_id')        
        wavelengths = np.transpose(pd.read_csv(data_dir() + 'wavelengths.csv').to_numpy())
        for i in range(len(data)):
            data[i]['planet_id'] = self.planet_ids_to_load[i]
            data[i]['is_train'] = self.load_train
            data[i]['star'] = adc_info['star'].loc[data[i]['planet_id']]
            data[i]['wavelengths_report'] = wavelengths

        if self.include_labels:
            assert self.load_train
            train_labels=pd.read_csv(data_dir() + 'train_labels.csv')
            train_labels = train_labels.set_index('planet_id')     
            for i in range(len(data)):
                data[i]['labels'] = train_labels.loc[data[i]['planet_id']].to_numpy()
        
        return data

def load_and_preprocess(planet_id, is_train, use_cache, loader_options):
    # Loads FGS and AIRS.

    # Inputs
    # planet_id: planet number to load
    # is_train: whether to load from the training set (rather than test)
    # use_cache: whether to try to get the data from cache; caching framework is not included in my shared Kaggle code, so keep this false
    # loader_options: LoaderOptionsBaseSignal 
    
    # Outputs a dict with FGS and AIRS info (see code for details...)

    # Use cache if possible
    if use_cache and is_train==1 and running_on_kaggle==0:
        loader_cache_info = pickle_load(file_loc()+'loader_cache.pickle')
        for i in range(len(loader_cache_info['options'])):
            if pickle.dumps(loader_options)==pickle.dumps(loader_cache_info['options'][i]):
                print('Used cache')
                return pickle_load(data_dir() + 'my_data/train_preprocessed_' + str(planet_id) + loader_cache_info['names'][i] + '.dat')

    # Determine which part of detector to use for AIRS ([r1:r2,r3:r3])
    r1 = loader_options.row_bottom
    r2 = loader_options.row_top
    r3 = 39
    r4 = 321
        
    # Determine wavelengths for AIRS
    wavelengths=axis_info['AIRS-CH0-axis2-um'].to_numpy()
    wavelengths=wavelengths[np.logical_not(np.isnan(wavelengths))]
    wavelengths=wavelengths[r3:r4]

    # Determine times for AIRS
    times=axis_info['AIRS-CH0-axis0-h'].to_numpy()
    times=times[np.logical_not(np.isnan(times))]
    times = times[0::2]/2+times[1::2]/2   

    # Load AIRS data
    data, t_ingress, t_egress, noise_est, points_per_bin, times = \
        load_and_preprocess_base_signal(planet_id, is_train, loader_options.options_AIRS, loader_options, times, r1,r2,r3,r4,-1,-1)   
    

    # Flip AIRS data for easier matching to wavelengths
    if not loader_options.options_AIRS.column_binning == 0:
        noise_est = np.flip(noise_est)
        data = np.flip(data,axis=1)
        wavelengths = np.flip(wavelengths)    
        if np.any(np.isnan(data)) or np.any(data==0):
            raise ArielException(5, 'NaN or zero in data')

    # Store AIRS data and ingress/egress time estimates
    output = dict()
    output['AIRS'] = dict()
    output['AIRS']['data'] = data
    output['AIRS']['wavelengths'] = wavelengths
    output['AIRS']['times'] = times
    output['AIRS']['noise_est'] = noise_est
    output['AIRS']['points_per_bin'] = points_per_bin
    output['t_ingress'] = t_ingress
    output['t_egress'] = t_egress

    if loader_options.load_FGS:
        # Determine FGS wavelengths
        wavelengths = np.reshape(pd.read_csv(data_dir() + 'wavelengths.csv').to_numpy()[0,0], (1,))
    
        # Determine FGS times
        times=axis_info['FGS1-axis0-h'].to_numpy()
        times = times[0::2]/2+times[1::2]/2   

        # Load FGS data
        data, t_ingress, t_egress, noise_est, points_per_bin, times = \
            load_and_preprocess_base_signal(planet_id, is_train, loader_options.options_FGS, loader_options, times, 0,32*32,0,1,t_ingress,t_egress)
    
        # Store FGS data
        output['FGS'] = dict()
        output['FGS']['data'] = data
        output['FGS']['wavelengths'] = wavelengths
        output['FGS']['times'] = times
        output['FGS']['noise_est'] = noise_est
        output['FGS']['points_per_bin'] = points_per_bin    
    
    return output


def load_and_preprocess_base_signal(planet_id, is_train, loader_options, loader_options_top, times,r1,r2,r3,r4,t_ingress,t_egress):
    # Load a single signal (FGS or AIRS)
    
    # Inputs
    # planet_id: planet number to load
    # is_train: whether to load from the training set (rather than test)
    # loader_options: LoaderOptionsBaseSignal
    # loader_options_top: LoaderOptions
    # times: timestamps for each frame
    # r1,r2,r3,r4: which part of the detector signal to use ([r1:2,r3:4])
    # t_ingress, t_egress: estimates for ingress and egress times; if set to -1 this function will detect them

    # Outputs data, t_ingress, t_egress, noise_est, points_per_bin, times
    # data: 2D or 3D array containing signals per pixel; first dimension is time
    # t_ingress, t_egress: estimated ingress and egress times
    # noise_est: estimated noise per pixel
    # points_per_bin: how many frames are combined for each timestamp in time binning
    # times: timestamps for each frame after time binning
    
    # functions below mainly come from example notebook, with small modifications
    def ADC_convert(signal, gain, offset):
        signal = signal.astype(np.float64)
        signal /= gain
        signal += loader_options.adc_offset_sign * offset
        sanity_check(lambda x:x, gain, 'gain', 1, [0.3, 3])                
        sanity_check(lambda x:x, offset, 'offset', 1, [-3000, 3000])
        return signal

    def mask_hot_dead(signal, dead, dark):
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
        hot = np.tile(hot, (signal.shape[0], 1, 1))
        dead = np.tile(dead, (signal.shape[0], 1, 1))
        if sanity_checks_active:
            sanity_check(lambda x:x, np.mean(hot), 'ratio_hot', 2, [0, 0.01])        
            sanity_check(lambda x:x, np.mean(dead), 'ratio_dead', 2, [0, 0.01])        
        if loader_options.mask_hot:
            signal[hot] = np.nan
        if loader_options.mask_dead:
            signal[dead] = np.nan
        return signal

    def apply_linear_corr(lc,x):        
        result =  lc[0,:,:]+x*(lc[1,:,:]+x*(lc[2,:,:]+x*(lc[3,:,:]+x*(lc[4,:,:]+x*lc[5,:,:]))))
        if sanity_checks_active:
            sanity_check(np.nanmax, np.abs(result-x), 'linear_corr_impact', 1, [0, 50000])
        return result
        
    def clean_dark(signal, dark, dt):    
        signal -= loader_options.dark_current_sign * dark * dt[:, np.newaxis, np.newaxis]
        sanity_check(np.min, gain, 'dark_min', 1, [0.3, 3]) 
        sanity_check(np.max, gain, 'dark_max', 1, [0.3, 3])        
        return signal

    def correct_flat_field(flat, signal):        
        flat = np.tile(flat, (signal.shape[0], 1, 1))
        signal = signal / flat
        sanity_check(np.min, flat, 'flat_min', 1, [-1, 1.5]) 
        sanity_check(np.max, flat, 'flat_max', 1, [0.7, 1.5])                
        return signal

    # Binning in time
    def bin_obs_loader(data, timelist):
        data_binned = np.zeros(data.shape)
        cur_pos_pre = 0
        cur_pos_post = 0
        points_per_bin = []
        # Create 1 frame at time in a loop
        while True:           
            if cur_pos_pre >= data.shape[0]:
                break
            cur_time = timelist[cur_pos_pre]

            # Figure out how many ponits to include, depening on if we're near ingress/egress
            if np.abs(cur_time-t_ingress)<loader_options.transition_margin or np.abs(cur_time-t_egress)<loader_options.transition_margin:
                this_time_bin = loader_options.time_binning_near_transitions
            else:
                this_time_bin = loader_options.time_binning                
            next_pos_pre = cur_pos_pre + this_time_bin
            if next_pos_pre>data.shape[0]:
                # Are we near the end? Then make a smaller bin with the remaining points.
                this_time_bin = data.shape[0] - cur_pos_pre
                next_pos_pre = cur_pos_pre + this_time_bin
            data_binned[cur_pos_post,:] = np.mean(data[cur_pos_pre:next_pos_pre,:], axis=0)
            cur_pos_post = cur_pos_post+1
            cur_pos_pre = next_pos_pre
            points_per_bin.append(this_time_bin)
        data_binned = data_binned[0:cur_pos_post,:]
        points_per_bin = np.array(points_per_bin)
        return data_binned, points_per_bin


    # Initial steps as in example notebook
    if is_train:
        train_str = 'train';
    else:
        train_str = 'test';
    dark = pd.read_parquet(data_dir()+train_str+'/'+str(planet_id)+'/'+loader_options.signal_name+'_calibration/dark.parquet').\
        values.astype(np.float64).reshape((loader_options.n_rows, loader_options.n_cols))[r1:r2,r3:r4]
    data = pd.read_parquet(data_dir()+train_str+'/'+str(planet_id)+'/'+loader_options.signal_name+'_signal.parquet')
    data = data.to_numpy().reshape(-1, loader_options.n_rows, loader_options.n_cols)[:,r1:r2,r3:r4];

    # ADC
    adc_info = pd.read_csv(data_dir() + train_str + '_adc_info.csv')
    adc_info = adc_info.set_index('planet_id')
    adc_info = adc_info.loc[planet_id]
    offset = adc_info[loader_options.signal_name+'_adc_offset']
    gain = adc_info[loader_options.signal_name+'_adc_gain']
    data = ADC_convert(data,gain,offset)

    # Mask
    dead = pd.read_parquet(data_dir()+train_str+'/'+str(planet_id)+'/'+loader_options.signal_name+'_calibration/dead.parquet').\
        values.astype(np.float64).reshape((loader_options.n_rows, loader_options.n_cols))[r1:r2,r3:r4]
    data = mask_hot_dead(data,dead>0,dark)

    # Select rows
    if loader_options.signal_name == 'AIRS-CH0':
        rows_to_use = np.arange(data.shape[1])
    else:
        if loader_options.n_pixels<np.inf:
            sums = np.sum(data,axis=(0,2))
            sums[np.isnan(sums)]=0
            inds = np.argsort(sums)
            rows_to_use = inds[-loader_options.n_pixels:]   
        else:
            rows_to_use = np.logical_not(np.isnan(np.sum(data,axis=(0,2))))        
    data = data[:,rows_to_use,:]
    dark = dark[rows_to_use,:]
    

    
    # Linear correction
    if loader_options.linear_correction:
        linear_corr = pd.read_parquet(data_dir()+train_str+'/'+str(planet_id)+'/'+loader_options.signal_name+'_calibration/linear_corr.parquet').\
            values.astype(np.float64).reshape((6, loader_options.n_rows, loader_options.n_cols))[:,r1:r2,r3:r4][:,rows_to_use,:]
        data = apply_linear_corr(linear_corr, data)
    
    # Dark current
    if loader_options.dark_current:
        if loader_options.signal_name == 'AIRS-CH0':
            dt = axis_info[loader_options.signal_name+'-integration_time'].dropna().values
        else:
            dt = np.ones(data.shape[0])*0.1
        dt[1::2] += 0.1
        data = clean_dark(data,dark,dt)

    # Flat field
    if loader_options.flat_field:
        flat = pd.read_parquet(data_dir()+train_str+'/'+str(planet_id)+'/'+loader_options.signal_name+'_calibration/flat.parquet').\
            values.astype(np.float64).reshape((loader_options.n_rows, loader_options.n_cols))[r1:r2,r3:r4][rows_to_use,:]
        data = correct_flat_field(flat, data)

    # Correlated double sampling    
    data = data[1::2,:,:]-data[0::2,:,:]

    # If a pixel is NaN at any point in time, it's always nan
    data = data+np.sum(data,axis=0)*0

    # Inpainting: fill invalid values usnig linear interpolation per row. Unhandled corner caes: two invalid neighbouring pixels the edge.
    if loader_options.inpainting:
        assert loader_options.signal_name == 'AIRS-CH0'
        # Interpolate the nans by row
        for y in range(data.shape[2]):
            for x in range(data.shape[1]):
                if np.isnan(data[0,x,y]):
                    # NOTE: not robust for double invalid edge pixels
                    if y==0:
                        data[:,x,y] = data[:,x,y+1]                        
                    elif y==data.shape[2]-1:
                        data[:,x,y] = data[:,x,y-1]
                    else:
                        ind_left = np.nonzero(np.logical_not(np.isnan(data[0,x,:y])))[0][-1]
                        ind_right= np.nonzero(np.logical_not(np.isnan(data[0,x,y+1:])))[0][0]+y+1
                        assert np.isnan(data[0,x,ind_left+1]) 
                        assert np.logical_not(np.isnan(data[0,x,ind_left]))
                        assert np.isnan(data[0,x,ind_right-1]) 
                        assert np.logical_not( np.isnan(data[0,x,ind_right]))
                        data[:,x,y] = 0.5*data[:,x,ind_left] + 0.5*data[:,x,ind_right]
        assert not np.any(np.isnan(data))

    # Background correction, as suggested by @cnumber on Kaggle. Actually this is called the 'foreground focal plane' for some reason.
    if loader_options.correct_background:
        if loader_options.signal_name == 'AIRS-CH0':
            # Note: hard-coded use of top 8 and bottom 8 rows for background correction, then remove these rows
            data_background=np.concatenate((data[:,0:8,:], data[:,24:32,:]), axis=1)
            background_estimate = np.nanmean(data_background,axis=(0,1))
            data = data - background_estimate[np.newaxis,np.newaxis,:]
            data = data[:,8:24,:];
            loader_options_top = copy.deepcopy(loader_options_top);
            loader_options_top.row_bottom = 8;
            loader_options_top.row_top = 24;
            r1 = 8; r2 = 24;
        else:
            data = data - loader_options.correct_background_fixed_fgs_value

    # Jitter correction
    if loader_options.correct_jitter:
        assert loader_options.signal_name == 'AIRS-CH0'
        data = correct_jitter(data, r1,r2, loader_options)        
        
    # Estimate ingress and egress time
    if t_ingress==-1:
        assert loader_options.signal_name == 'AIRS-CH0', "Ingress/egress detection on FGS is unreliable"
        ind_ingress, ind_egress = find_transit_window(np.nansum(data,axis=1))
        t_ingress = times[ind_ingress]; t_egress = times[ind_egress];   

    sanity_check(lambda x:x, t_ingress, 't_ingress_initial', 3, [1,3.5]) 
    sanity_check(lambda x:x, t_egress, 't_egress_initial', 4, [3.5,6.5])

    # Binning columns
    if loader_options.column_binning == 1:
        if loader_options.signal_name == 'AIRS-CH0':
            mean_target = np.nansum(data,axis=(1,2))
            data,_ = my_weighted_binning(data, loader_options, loader_options_top)        
            if loader_options.fix_mean_per_frame_in_column_binning:
                # Make sure we get the same mean per frame as with column_binning = -1
                mean_new = np.nansum(data,axis=1)
                data = data * (mean_target/mean_new)[:,np.newaxis]
        else:
            data = my_weighted_binning_fgs(data, loader_options, loader_options_top)
    elif loader_options.column_binning == -1:
        data = np.nansum(data, axis=1)
    else:
        assert loader_options.column_binning == 0

    # Estimate noise
    if loader_options.column_binning == 0:
        n_avg = loader_options.width_for_noise_estimate_ma
        noise_est = np.std(data[(n_avg-1)//2:-(n_avg-1)//2,:,:] - moving_average_3d(data,n_avg),axis=0)/np.sqrt(1-1/(n_avg-1))
    else:
        n_avg = loader_options.width_for_noise_estimate_ma
        noise_est = np.std(data[(n_avg-1)//2:-(n_avg-1)//2,:] - moving_average_2d(data,n_avg),axis=0)/np.sqrt(1-1/(n_avg-1))
    
    # Binning in time
    if loader_options.time_binning>0:
        data,_ = bin_obs_loader(data, times)
        times,points_per_bin = bin_obs_loader(np.reshape(times, (-1,1)), copy.deepcopy(times))
        times = times.flatten()
    else:
        points_per_bin = 0*times+1

    return data, t_ingress, t_egress, noise_est, points_per_bin, times

def find_transit_window(data, visualize=False, Navg=250, Noffset=150, fit_order=5):
    # Gives a rough estimate of the ingress and egress locations, using the following steps:
    # - Apply a moving average filter
    # - Remove a polynomial fit
    # - Find the index where the biggest increase/decrease occurs over a certain interval
    data_summed = np.sum(data,axis=1);
    data_smoothed = moving_average(data_summed, Navg)
    data_smoothed = data_smoothed-np.poly1d(np.polyfit(np.arange(len(data_smoothed)), data_smoothed, fit_order))(np.arange(len(data_smoothed)))
    
    diff = data_smoothed[Noffset:]-data_smoothed[:-Noffset]
    ind_ingress = int(np.argmin(diff)+(Navg/2)+(Noffset/2))
    ind_egress = int(np.argmax(diff)+(Navg/2)+(Noffset/2))
    if visualize:
       plt.figure()
       plt.plot(data_summed-np.mean(data_summed))
       plt.plot(Navg/2+np.arange(len(data_smoothed)), data_smoothed-np.mean(data_smoothed))
       plt.axline((ind_egress,0), slope=np.inf)
       plt.axline((ind_ingress,0), slope=np.inf)
    return ind_ingress, ind_egress

def my_weighted_binning(data, loader_options, loader_options_top):
    # Performs weighted binning on data, using weights such that:
    # -The mean of the data is correctly summed, i.e. sum(w*m)=sum(m) for every pixel, with m the mean over time.
    # -The impact of noise is minimized.
    # I didn't comment this one in detail because it's not used in the standard submission.   

    jitter_shapes = pickle_load(file_loc()+"jitter_shapes.pickle")
    
    n_avg = loader_options.width_for_noise_estimate_ma  

    mean_shape = np.reshape(jitter_shapes['mean'], jitter_shapes['shape'])[loader_options_top.row_bottom-8:loader_options_top.row_top-8,0:321-39]
    data_mean = mean_shape
         
    #data_mean = np.mean(data, axis=0)
    noise = data[(n_avg-1)//2:-(n_avg-1)//2,:,:] - moving_average_3d(data,n_avg)
    noise_scaled = noise/data_mean
    data_scaled = data/data_mean
    
    sum_est = np.zeros((data.shape[0], data.shape[2]))
    var_est = np.zeros((data.shape[2]))
    for i in range(data.shape[2]):
        this_column = noise_scaled[:,:,i];
        not_nan = np.logical_not(np.isnan(this_column[0,:]))
        this_column = this_column[:,not_nan]
        ones = np.ones((this_column.shape[1], 1))  
        cov_matrix_est_scaled = np.transpose(this_column)@this_column/(noise.shape[0]-1)
        interm = np.linalg.solve(cov_matrix_est_scaled, ones)
        var_est_scaled = 1/(np.transpose(ones)@interm)
        mean_est_scaled = (data_scaled[:,not_nan,i]@interm)*var_est_scaled
        sum_est[:,i] = mean_est_scaled[:,0]*np.sum(data_mean[:,i])
            
    return sum_est, 0*np.sqrt(var_est)

def my_weighted_binning_fgs(data, loader_options, loader_options_top):
    # Similar to function above, but for FGS.
    # I didn't comment this one in detail because it's not used in the standard submission.   

    data = np.reshape(data, (data.shape[0], -1))
 
    n_avg = loader_options.width_for_noise_estimate_ma
    data_mean = np.mean(data,axis=0)
    noise = data[(n_avg-1)//2:-(n_avg-1)//2,:] - moving_average_2d(data,n_avg)
    noise_scaled = noise/data_mean
    data_scaled = data/data_mean
    
    sum_est = np.zeros((data.shape[0], data.shape[1]))
    var_est = np.zeros((data.shape[1]))
    this_column = data;
    ones = np.ones((this_column.shape[1], 1))  
    cov_matrix_est_scaled = np.transpose(noise_scaled)@noise_scaled/(noise_scaled.shape[0]-1)
    interm = np.linalg.solve(cov_matrix_est_scaled, ones)
    var_est_scaled = 1/(np.transpose(ones)@interm)
    mean_est_scaled = (data_scaled@interm)*var_est_scaled
    return mean_est_scaled*np.sum(data_mean)

def correct_jitter(data, row_bottom, row_top, loader_options):
    # Corrects for the way the signal is distrubted over rows can drift over time. 
    # I didn't comment this one in detail because it's not used in the standard submission.  

    jitter_shapes = pickle_load(file_loc()+"jitter_shapes.pickle")

    cols_remove = np.any(np.isnan(data), axis=(0,1))
    mean_shape = np.reshape(jitter_shapes['mean'], jitter_shapes['shape'])[row_bottom-8:row_top-8,0:321-39]
    mean_shape_full = copy.deepcopy(mean_shape)
    mean_shape = mean_shape[:,np.logical_not(cols_remove)]
    mean_shape = mean_shape.flatten()
    
    pca_shapes = np.reshape(jitter_shapes['pca'].components_, (jitter_shapes['pca'].components_.shape[0], jitter_shapes['shape'][0], jitter_shapes['shape'][1]))\
        [:, row_bottom-8:row_top-8,0:321-39]
    pca_shapes_full = copy.deepcopy(pca_shapes)
    pca_shapes = pca_shapes[:,:,np.logical_not(cols_remove)]
    pca_shapes = np.reshape(pca_shapes, (jitter_shapes['pca'].components_.shape[0], -1))    
    
    data_ref = data[:,:,np.logical_not(cols_remove)]
    data_ref = data_ref/np.sum(data_ref,axis=1)[:,np.newaxis,:]
    orig_shape = data_ref.shape
    data_ref = np.reshape(data_ref, (data_ref.shape[0], -1))
    data_ref = data_ref-mean_shape
    
    x=np.linalg.lstsq((pca_shapes).T, (data_ref).T, rcond=None)[0]

    pred = (pca_shapes_full.T@x).T
    pred = np.reshape(pred, data.shape)
    
    data = data/(1 + pred/mean_shape_full)

    return data

'''
Modeling framework

Defines the abstract class for modeling, as well as some helper models. The 'real' model is in ariel_gp.py.
'''
class Model:
    # General modeling class
    # Interface: first run train giving the training data, then you can run infer using the test data
    # This is an abstract class, subclasses must implement:
    # -train_internal
    # -infer_internal OR infer_internal_single
    
    is_trained = False # set to True by train()
    run_in_parallel = True # whether to use parallel pool
    
    def train(self, train_data):
        assert not self.is_trained
        self.train_internal(train_data) # to be implemented by subclass
        self.is_trained = True
        
    def infer(self, test_data):
        # Model inference. Returns:
        # pred: predicted transit depth
        # sigma: confidence interval
        # cov: full covariance matrix, sigma should be the square root of the diagonal of cov
        assert self.is_trained
        pred,sigma,cov = self.infer_internal(test_data) # to be implemented by subclass
        pred[pred<0]=0 # make legal
        assert np.all(sigma>0)
        if np.any(np.isnan(sigma)) or np.any(np.isnan(pred)):
            raise ArielException(5, 'Pred or sigma contains NaN')        
        return pred,sigma,cov
    
    def infer_internal(self, test_data):
        # This is a possible implementation of infer_internal that loops infer_internal_single over all planets. Subclasses may provide an alternative implementation. If they do not they must implement infer_internal_single.
        if (not self.run_in_parallel) or (not multiprocessing.current_process().name=='MainProcess') or (running_on_kaggle==1) or (never_model_parallel):
            pred,sigma,cov = zip(*[self.infer_internal_single(d) for d in test_data])
        else:
            p = multiprocessing.Pool(recommend_n_workers())
            pred,sigma,cov = zip(*  p.starmap(infer_internal_single_parallel, zip(test_data, itertools.repeat(self))))
            p.close()
        pred = np.stack(pred)
        sigma = np.stack(sigma)      
        cov = list(cov)
        return pred,sigma,cov

# Function is used above, I ran into issues with multiprocessing if it was not a top-level function
def infer_internal_single_parallel(data, model):
    try:
        return model.infer_internal_single(data)
    except Exception as err:
        import traceback
        print('Error for planet: ' + str(data['planet_id']))
        print(traceback.format_exc())     
        raise

class SigmaFudger(Model):
    # Multiplies all sigma values of another model such that the training score is optimized
    model = None # The actual model
    fudge_value = 1. # Set while training

    # Cached results of the internal model; this speeds things up if we do inference on the same set
    pred_cached = None
    sigma_cached = None
    input_cached = None
    
    def train_internal(self, train_data):
        # Infer internal model on training set
        self.model.train(train_data)
        pred,sigma,cov = self.model.infer(train_data)

        # Cache results
        self.pred_cached = pred
        self.sigma_cached = sigma
        self.cov_cached = cov
        self.input_cached = hashlib.sha256(pickle.dumps(train_data)).digest()

        # Find fudge_value to optimize score
        planet_list = [x['planet_id'] for x in train_data]
        def score_fudged(fudge_value):
            return -score_wrapper(planet_list, pred, np.exp(fudge_value)*sigma)[0]
        fudge_value = np.exp(sp.optimize.fmin_l_bfgs_b(score_fudged, 0, approx_grad=True, bounds = [(-3, 3)])[0])
        self.fudge_value = fudge_value

        # Report
        print('Chosen sigma fudge: '+str(self.fudge_value)+' improving score from ' + str(-score_fudged(0)) + ' to ' + str(-score_fudged(np.log(fudge_value))))

    def infer_internal(self, test_data):
        
        # Infer internal model, using cache if possible
        if (not (self.input_cached) is None) and (hashlib.sha256(pickle.dumps(test_data)).digest()==self.input_cached):
            pred,sigma,cov= self.pred_cached,self.sigma_cached,self.cov_cached
        else:
            pred,sigma,cov = self.model.infer(test_data)

        # Adjust sigma and covariance matrix
        cov = copy.deepcopy(cov) # maybe not needed? don't dare to mess with it
        for i in range(len(cov)):
            cov[i] = self.fudge_value**2 * cov[i]
        return pred,self.fudge_value*sigma,cov

class MeanBiasFitter(Model):
    # Similar to above, but applies a scaling to the mean of each prediction rather than to the sigma values. Most of the code is duplicated; see above for comments.
    model = None
    bias = 0.
    pred_cached = None
    sigma_cached = None
    input_cached = None
    def train_internal(self, train_data):
        self.model.train(train_data)
        pred,sigma,cov = self.model.infer(train_data)
        self.pred_cached = pred
        self.sigma_cached = sigma
        self.cov_cached = cov
        self.input_cached = hashlib.sha256(pickle.dumps(train_data)).digest()
        
        true_mean = [np.mean(d['labels']) for d in train_data]
        pred_mean = np.mean(pred, axis=1)
        mean_error = pred_mean - true_mean
        self.bias = -np.linalg.lstsq(np.reshape(true_mean, (-1,1)), np.reshape(mean_error, (-1,1)), rcond=-1)[0]
        
        print('Chosen bias: '+str(self.bias))

    def infer_internal(self, test_data):
        if (not (self.input_cached) is None) and (hashlib.sha256(pickle.dumps(test_data)).digest()==self.input_cached):
            pred,sigma,cov= self.pred_cached,self.sigma_cached,self.cov_cached
        else:
            pred,sigma,cov = self.model.infer(test_data)
        pred = copy.deepcopy(pred)
        for i in range(len(pred)):
            pred[i] = pred[i] + np.mean(pred[i])*self.bias
        return pred,sigma,cov

'''
Sanity checks

This section defines infrastructure to make sure that nothing weird is happening on the private test planets. You can use the sanity_check function in code elsewhere to track certain values, and compare them to thresholds. You can find the thresholds by running code on the training data without using parallel pool, and then running print_sanity_checks.
'''
# Class to keep track of sanity checks
class SanityCheckValue:
    def __init__(self, name, code, limit):
        self.seen = [np.inf, -np.inf]
        self.limit = limit
        self.name = name
        self.code = code

# Perform a sanity check; this function is used throughout the data loading and modeling functions.
def sanity_check(f,to_check,name,code,limit):
    # f: function to apply to to_check
    # to_check: value to check
    # name: label for the sanity check
    # code: error code number to give
    # limit: 2-element array containing lower and upper limit
    if sanity_checks_active:
        if not name in sanity_checks:
            sanity_checks[name] = SanityCheckValue(name,code,limit)
        value = float(f(to_check))
        if not sanity_checks_without_errors:
            if value > limit[1]:
                raise ArielException(code+0.5,name + ' too high: ' + str(value) + '>' + str(limit[1]))
            if value < limit[0]:
                raise ArielException(code,name + ' too low: ' + str(value) + '<' + str(limit[0]))
        # Keep track of extreme values seen (helps in determining thresholds)
        if sanity_checks[name].seen[0] > value:
            sanity_checks[name].seen[0] = value
        if sanity_checks[name].seen[1] < value:
            sanity_checks[name].seen[1] = value

# Show the minimum and miximum values seen for all sanity checks
def print_sanity_checks():
    for m in list(sanity_checks.keys()):
        print(m, sanity_checks[m].seen, sanity_checks[m].limit, sanity_checks[m].code)

class ArielException(Exception):
    # Exception class for failed sanity checks
    code = 0
    def __init__(self,code,message):
        self.code = code
        self.message = message


'''
Scoring functions
'''    
  
def score_wrapper(planet_list,pred,sigma):
    # Wrapper to convert pred and sigma as obtained from a model into the Kaggle score and RMS
    submission = pd.read_csv(data_dir() + '/sample_submission.csv')
    submission = submission[0:0]
    
    for i in range(len(planet_list)):
        submission.loc[i] = np.concatenate(([planet_list[i]], pred[i], sigma[i]))

    train_labels=pd.read_csv(data_dir() + 'train_labels.csv')
    train_labels = train_labels.set_index('planet_id')
    train_labels = train_labels.loc[planet_list]
    train_labels = train_labels.reset_index()
    train_labels_np = train_labels.drop('planet_id', axis=1).to_numpy()
    error = train_labels_np - submission.to_numpy()[:,1:284]
    rms_error = np.sqrt(np.mean(np.sum(error*error)/(error.shape[0]*error.shape[1])))
    train_labels_full=pd.read_csv(data_dir() + 'train_labels.csv').set_index('planet_id').to_numpy()
    score_val = score(train_labels, submission, "planet_id", np.mean(train_labels_full), np.std(train_labels_full), 1e-5)
    return score_val, rms_error

# Scoring function as defined by organizers
def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        naive_mean: float,
        naive_sigma: float,
        sigma_true: float
    ) -> float:
    '''
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise Exception('Negative values in the submission')
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise Exception(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths*2:
        raise Exception('Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    y_true = solution.values

    GLL_pred = np.mean(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
    GLL_true = np.mean(scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)))
    GLL_mean = np.mean(scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true)))

    submit_score = (GLL_pred - GLL_mean)/(GLL_true - GLL_mean)
    return float(submit_score) # I removed the clipping to 0 or 1    