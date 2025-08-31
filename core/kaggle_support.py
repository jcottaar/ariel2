import pandas as pd
import numpy as np
import scipy as sp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython
from dataclasses import dataclass, field, fields
import enum
import typing
import pathlib
import multiprocess
multiprocess.set_start_method('spawn', force=True)
from decorator import decorator
from line_profiler import LineProfiler
import os
import gc
import glob
import h5py
import time
import sklearn
import shutil
import torch
import inspect
from tqdm import tqdm
import hashlib


'''
Determine environment and globals
'''

if os.path.isdir('/mnt/d/ariel2/'):
    env = 'local'
    d_drive = '/mnt/d/'    
elif os.path.isdir('d:/'):
    env = 'local_windows'
    d_drive = 'd:/'   
else:
    env = 'kaggle'
print(env)

profiling = False
debugging_mode = 2
verbosity = 1
disable_any_parallel = False

match env:
    case 'local' | 'local_windows':
        data_dir = d_drive+'/ariel2/data/'
        temp_dir = d_drive+'/ariel2/temp/'             
        code_dir = d_drive+'/ariel2/code/core/' 
        csv_dir = d_drive+'/ariel2/'
        calibration_dir = d_drive+'/ariel2/calibration/'
        loader_cache_dir = d_drive+'/ariel2/loader_cache/'
    case 'kaggle':
        data_dir = '/kaggle/input/ariel-data-challenge-2025/'
        temp_dir = '/temp/'           
        loader_cache_dir = '/temp/loader_cache/'
        code_dir = '/kaggle/input/my-ariel2-library/'         
        csv_dir = '/kaggle/working/'
        calibration_dir = '/kaggle/input/my-ariel2-calibration/'
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(loader_cache_dir, exist_ok=True)

# How many workers is optimal for parallel pool?
n_workers = 3 if env=='kaggle' else 6
def recommend_n_workers():
    return n_workers
n_threads = 1

# os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
# os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")           # disable TF32 (TF32 can vary)
# os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")     # more stable kernel ordering
n_cuda_devices = torch.cuda.device_count()
process_name = multiprocess.current_process().name
if not multiprocess.current_process().name == "MainProcess":
    print(process_name, multiprocess.current_process()._identity[0])  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.mod(multiprocess.current_process()._identity[0], n_cuda_devices))
    print('CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]);

import cupy as cp

if not env=='kaggle':
    import git 
    repo = git.Repo(search_parent_directories=True)
    git_commit_id = repo.head.object.hexsha
else:
    git_commit_id = 'kaggle'


'''
Helper classes and functions
'''

def list_attrs(obj):
    for name, val in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        # # skip methods, but let descriptors through
        # if callable(val) and not isinstance(val, property):
        #     continue
        print(f"{name} = {val}")

def remove_and_make_dir(path):
    try: shutil.rmtree(path)
    except: pass
    os.makedirs(path)

# Helper class - doesn't allow new properties after construction, and enforces property types. Partially written by ChatGPT.
@dataclass
class BaseClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)
    comment:str = field(init=True, default='')

    def check_constraints(self, debugging_mode_offset = 0):
        global debugging_mode
        debugging_mode = debugging_mode+debugging_mode_offset
        try:
            if debugging_mode > 0:
                self._check_types()
                self._check_constraints()
            return
        finally:
            debugging_mode = debugging_mode - debugging_mode_offset

    def _check_constraints(self):
        pass

    def _check_types(self):
        type_hints = typing.get_type_hints(self.__class__)
        for field_info in fields(self):
            field_name = field_info.name
            expected_type = type_hints.get(field_name)
            actual_value = getattr(self, field_name)
            
            if expected_type and not isinstance(actual_value, expected_type) and not actual_value is None:
                raise TypeError(
                    f"Field '{field_name}' expected type {expected_type}, "
                    f"but got value {actual_value} of type {type(actual_value).__name__}.")

    def __post_init__(self):
        # Mark the object as frozen after initialization
        object.__setattr__(self, '_is_frozen', True)

    def __setattr__(self, key, value):
        # If the object is frozen, prevent setting new attributes
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)

# Small wrapper for dill loading
def dill_load(filename):
    filehandler = open(filename, 'rb');
    data = dill.load(filehandler)
    filehandler.close()
    return data

# Small wrapper for dill saving
def dill_save(filename, data):
    filehandler = open(filename, 'wb');
    data = dill.dump(data, filehandler)
    filehandler.close()
    return data

@decorator
def profile_each_line(func, *args, **kwargs):
    if not profiling:
        return func(*args, **kwargs)
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        s=profiled_func(*args, **kwargs)
        profiler.print_stats()
        return s
    except:
        profiler.print_stats()
        raise

def profile_print(string):
    if profiling: print(string)

def download_kaggle_dataset(dataset_name, destination, skip_download=False):
    remove_and_make_dir(destination)
    if not skip_download:
        subprocess.run('kaggle datasets download ' + dataset_name + ' --unzip -p ' + destination, shell=True)
    subprocess.run('kaggle datasets metadata -p ' + destination + ' ' + dataset_name, shell=True)

def upload_kaggle_dataset(source):
    if env=='local_windows':
        source=source.replace('/', '\\')
    subprocess.run('kaggle datasets version -p ' + source + ' -m ''Update''', shell=True)

def rms(array):
    return np.sqrt(np.mean(array**2))

def ismembertol(a,b,reltol=1e-4):
    # Returns array x of length len(a), subh that b[x]=a. Entries of a that are not in b get value -1. Not particularly efficient.
    b_unique,inds_unique = np.unique(b, return_index = True)
    def find_element(el):
        is_close = np.abs(el-b_unique) < reltol*np.abs(el+b_unique)
        if np.any(is_close):
            return inds_unique[np.argwhere(is_close)[0,0]]
        else:
            return -1
    return np.array([find_element(x) for x in a])


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:, ...] -= ret[:-n, ...]
    return ret[n - 1:, ...] / n

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

def add_cursor(sc):

    import mplcursors
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"index={i}")


def to_cpu(array):
    if isinstance(array, cp.ndarray):
        return array.get()
    else:
        return array

def to_gpu(array):
    if isinstance(array, cp.ndarray):
        return array
    else:
        return cp.array(array)
    
def clear_gpu():
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    cp.get_default_memory_pool().free_all_blocks()
'''
Sanity checks and error handling
'''

class ArielException(Exception):
    # Exception class for failed sanity checks
    code = 0
    def __init__(self,code,message):
        self.code = code
        self.message = message

score_base = 0.264
score_noise = 0.121
noise_fac_used = 1.5
error_code_conversion = 5
def raise_error_code(exception):    
    if not env=='kaggle':
        raise exception
    print('ARIEL exception ', exception.code, exception.message)
    assert(exception.code>=0)
    noise_fac = np.sqrt(exception.code/error_code_conversion)
    dill_save(temp_dir + '/noise_fac.pickle', noise_fac)
    import subprocess
    subprocess.run('jupyter nbconvert --execute --debug --to html /kaggle/input/my-ariel2-library/reference_submission.ipynb', shell=True)
    
def error_code_from_score(score):
    return (score_base-score)/(score_base-score_noise)*noise_fac_used**2*error_code_conversion

sanity_checks_active = True # Whether to throw errors if we see weird things in the data
sanity_checks_without_errors = False # Perform the sanity checks above but don't throw errors
sanity_checks = dict() # Global object to keep track of all sanity checks. 
    
# Class to keep track of sanity checks
class SanityCheckValue:
    def __init__(self, name, code, limit):
        self.seen = [np.inf, -np.inf]
        self.limit = limit
        self.name = name
        self.code = code
        self.seen_all = []

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
        value = float(to_cpu(f(to_check)))
        if not sanity_checks_without_errors:
            if value > limit[1]:
                raise ArielException(code+0.5,name + ' too high: ' + str(value) + '>' + str(limit[1]))
            if value < limit[0]:
                raise ArielException(code,name + ' too low: ' + str(value) + '<' + str(limit[0]))
        # Keep track of extreme values seen (helps in determining thresholds)
        sanity_checks[name].seen_all.append(value)
        if sanity_checks[name].seen[0] > value:
            sanity_checks[name].seen[0] = value
        if sanity_checks[name].seen[1] < value:
            sanity_checks[name].seen[1] = value
        

# Show the minimum and miximum values seen for all sanity checks
def print_sanity_checks():
    for m in list(sanity_checks.keys()):
        print(m, sanity_checks[m].seen, sanity_checks[m].limit, sanity_checks[m].code)
        
def plot_sanity_checks():
    for k,v in sanity_checks.items():
        plt.figure()
        plt.grid(True)
        plt.plot(v.seen_all)
        plt.axline((0,v.limit[0]), slope=0,color='red')
        plt.axline((0,v.limit[1]), slope=0,color='red')
        plt.title(k)

'''
Data definition and loading
'''

# Some input data that we only have to load once
axis_info = pd.read_parquet(data_dir+'axis_info.parquet')
train_star_info = pd.read_csv(data_dir+'train_star_info.csv')
test_star_info = pd.read_csv(data_dir+'test_star_info.csv')
train_labels = pd.read_csv(data_dir+'train.csv')
wavelengths = pd.read_csv(data_dir+'wavelengths.csv').to_numpy()[0,:]
wavelengths_cp = cp.array(wavelengths)
adc_info = pd.read_csv(data_dir+'adc_info.csv')
sensor_names = ['FGS1', 'AIRS-CH0']

@dataclass
class Planet(BaseClass):
    # Holds loaded or estimated planet properties (atmospheric, stellar, and orbital), as well as the transit measurements
    planet_id: int = field(init=True, default=None) # Unique identifier for the star-planet system.
    is_train: bool = field(init=True, default=None) # Whether this is a train or test planet.
    transit_params: object = field(init=True, default=None)
    
    spectrum: np.ndarray = field(init=True, default=None) # (Rp/Rs)^2, can be None or 1D array with lenght of wavelengths
    spectrum_cov: np.ndarray = field(init=True, default=None) # covariance matrix for uncertainty in spectrum, can be None or 2D array
    
    transits: list = field(init=True, default_factory = list) # transit observations for this planet
    
    diagnostics: dict = field(init=True, default_factory = dict)
    
    def __post_init__(self):
        super().__post_init__()
        import ariel_transit
        self.transit_params = ariel_transit.TransitParams()        

    
    def _check_constraints(self):
        if not self.spectrum is None:
            assert self.spectrum.shape == wavelengths.shape
        if not self.spectrum_cov is None:
            assert self.spectrum_cov.shape == (wavelengths.shape[0], wavelengths.shape[0])
        assert len(self.transits)>=1
        for t in self.transits:
            assert isinstance(t, Transit)
            t.check_constraints()
    
    def load_orbital_and_stellar_properties(self):
        if self.is_train:
            row = train_star_info[train_star_info['planet_id']==self.planet_id]
        else:
            row = test_star_info[test_star_info['planet_id']==self.planet_id]
        assert row.shape[0]==1
        row = row.iloc[0]        
        self.transit_params.Rs = row['Rs']
        self.transit_params.Ms = row['Ms']
        self.transit_params.Ts = row['Ts']
        self.transit_params.Mp = row['Mp']
        self.transit_params.e = row['e']
        self.transit_params.P = row['P']*24
        self.transit_params.sma = row['sma']
        self.transit_params.i = row['i']
        assert self.transit_params.e==0.
        
    def load_spectrum(self):
        assert self.is_train
        self.spectrum = train_labels[train_star_info['planet_id']==self.planet_id].to_numpy()
        assert self.spectrum.shape[0]==1    
        self.spectrum = self.spectrum[0,1:]
        self.spectrum_cov=None        
        self.diagnostics['training_spectrum'] = copy.deepcopy(self.spectrum)
    
    def unload_spectrum(self):
        self.spectrum = None
        self.spectrum_cov = None
        
    def get_directory(self):
        if self.is_train:
            return data_dir + '/train/' + str(self.planet_id) + '/'
        else:
            return data_dir + '/test/' + str(self.planet_id) + '/'
        
    def load_to_step(self, target_step, loaders):
        for t in self.transits:
            t.load_to_step(target_step, self, loaders)
        
@dataclass
class Transit(BaseClass):
    # Holds all relevant information for a single transit    
    observation_number: int = field(init=True, default=None) # how manieth transit is this
    loading_step: int = field(init=True, default=0) # internal state
    # 0: unloaded
    # 1: raw parquet loaded
    # 2: pixel level corrections up to correlated double sampling done
    # 3: full-sensor corrections done
    # 4: time binning done
    # 5: wavelength binning done
    data: list = field(init=True, default_factory = lambda:[SensorData(is_FGS=True), SensorData(is_FGS=False)]) # 0: FGS, 1: AIRS
    
    diagnostics: dict = field(init=True, default_factory = dict)
    
    def _check_constraints(self):
        assert len(self.data)==2
        assert self.data[0].is_FGS
        assert not self.data[1].is_FGS
        for d in self.data:
            assert(isinstance(d, SensorData))
            assert d.loading_step == self.loading_step
            d.check_constraints()
    
    def load_to_step(self, target_step, planet, loaders):
        self.check_constraints()
        if target_step == self.loading_step:
            return
        caching = target_step in loaders[0].cache_steps
        if caching:
            cache_file_name = loader_cache_dir + '/' + hashlib.sha256(dill.dumps(loaders)).hexdigest()[:10] + '_' + str(planet.planet_id) + '_' + str(self.observation_number) + '_' + str(planet.is_train) + '_' + str(target_step) +'.pickle';
            #print(cache_file_name)
            if os.path.isfile(cache_file_name):
                (self.data, self.diagnostics) = dill_load(cache_file_name)
                self.loading_step = target_step
                self.check_constraints()
                return
            #cache_file_name = loader_cache_dir + '/' + 
        if target_step<self.loading_step:
            self.loading_step = 0
            self.data = Transit().data
            self.check_constraints()
            self.load_to_step(target_step, planet, loaders)
            return
        if target_step>self.loading_step+1:
            self.load_to_step(target_step-1, planet, loaders)
            assert target_step==self.loading_step+1
        if target_step==self.loading_step+1:
            for loader in loaders:
                assert isinstance(loader, TransitLoader)
                loader.check_constraints()                
            for d,loader in zip(self.data, loaders):
                loader.progress_one_step(d, planet, self.observation_number)
            self.loading_step += 1
            self.check_constraints()
        if caching:
            assert not os.path.isfile(cache_file_name)
            dill_save(cache_file_name, (self.data, self.diagnostics))
        assert self.loading_step == target_step
        
@dataclass
class SensorData(BaseClass):
    # Holds data for one sensor and one transit
    is_FGS: bool = field(init=True, default=None)
    loading_step: int = field(init=True, default=0) # internal state, see Transit
    
    data: cp.ndarray = field(init=True, default=None) 
    # 1st dimension: time
    # 2nd dimension: wavelength, but only if in step 5
    # further dimensions (0, 1, or 2): sensor coordinates
    times: cp.ndarray = field(init=True, default=None) # time associated with dimension 1 above, in seconds
    time_intervals: cp.ndarray = field(init=True, default=None) # time integration lengths, same size as above, in seconds
    wavelengths: cp.ndarray = field(init=True, default=None) # wavelengths associated with dimension 2 above, in um
    noise_est: cp.ndarray = field(init=True, default=None) # 1 sigma noise estimate per pixel, normalized to integration time of 1 second
        
    
    def _check_constraints(self):
        if self.loading_step==0:
            assert self.data is None
        elif self.loading_step<=4:
            assert len(self.data.shape)==3
        else:
            assert len(self.data.shape)==2
            
        if self.loading_step>=1:
            assert(self.times.shape == (self.data.shape[0],))
            assert(self.time_intervals.shape == (self.data.shape[0],))
        
        if self.loading_step==5:
            assert(self.wavelengths.shape == (self.data.shape[1],))            
            assert(self.noise_est.shape == (self.data.shape[1],))        
        elif self.loading_step>=1 and not self.is_FGS:
            assert(self.wavelengths.shape == (self.data.shape[2],))    
        
@dataclass
class TransitLoader(BaseClass):
    # Manages loading transit data. User must fill in the classes below.
    
    cache_steps: list = field(init=True, default_factory = lambda:[5])
    
    load_raw_data: BaseClass = field(init=True, default=None) # must be callable
    apply_pixel_corrections: BaseClass = field(init=True, default=None) # must be callable
    apply_full_sensor_corrections: BaseClass = field(init=True, default=None) # must be callable
    apply_time_binning: BaseClass = field(init=True, default=None) # must be callable
    apply_wavelength_binning: BaseClass = field(init=True, default=None) # must be callable
    
    def progress_one_step(self, data, planet, observation_number):
        match data.loading_step:
            case 0:
                self.load_raw_data(data, planet, observation_number)
            case 1:
                self.apply_pixel_corrections(data, planet, observation_number)
            case 2:
                self.apply_time_binning(data, planet, observation_number)
            case 3:
                self.apply_full_sensor_corrections(data, planet, observation_number)            
            case 4:
                self.apply_wavelength_binning(data, planet, observation_number)
            case _:
                raise Exception('Wrong state')
        data.loading_step+=1
            

def load_data(planet_id, is_train):
    data = Planet()
    data.planet_id = planet_id
    data.is_train = is_train
    
    data.load_orbital_and_stellar_properties()    
    if is_train:
        data.load_spectrum()
    else:
        data.diagnostics['training_spectrum'] = None
        
    files = glob.glob(data.get_directory()+'*')
    assert len(files)%4 == 0
    n_transits = len(files)//4
    sanity_check(lambda x:x, n_transits, 'n_transits', 1, [1,2])
    for ii in range(n_transits):
        data.transits.append(Transit())
        data.transits[-1].observation_number = ii
    
    data.check_constraints()
    
    return data

def load_all_train_data():
    return [load_data(ind,True) for ind in train_star_info['planet_id'] if ind!=2486733311]

def load_all_test_data():
    return [load_data(int(ind),False) for ind in test_star_info['planet_id']]

    
'''
General model definition
'''
# Function is used below, I ran into issues with multiprocessing if it was not a top-level function
model_parallel = None
def infer_internal_single_parallel(data):   
    try:        
        global model_parallel
        global sanity_checks_active
        global sanity_checks_without_errors
        global debugging_mode
        global profiling
        global n_threads
        global sanity_checks
        if model_parallel is None:
            model_parallel, sanity_checks_active, sanity_checks_without_errors, debugging_mode, profiling, n_threads = dill_load(temp_dir+'parallel.pickle')
        sanity_checks = dict()
        t=time.time()
        orig_step = data.transits[0].loading_step
        from threadpoolctl import threadpool_limits
        from contextlib import nullcontext
        if n_threads==np.inf:
            env = nullcontext()
        else:
            env = threadpool_limits(limits=n_threads)                 
        with env:
            data = model_parallel._infer_single(data)
        data.load_to_step(orig_step,model_parallel.loaders)        
        data.diagnostics['sanity_checks_par'] = sanity_checks
        return data
    except Exception as err:
        import traceback
        print(traceback.format_exc())     
        print('Planet ID: ', data.planet_id)
        dill_save(temp_dir+'error.pickle', (err, data.planet_id))
        raise


@dataclass
class Model(BaseClass):
    # Loads one or more cryoET measuerements
    state: int = field(init=False, default=0) # 0: untrained, 1: trained    
    run_in_parallel: bool = field(init=True, default=False) 

    loaders: list = field(init=True, default=None)
    
    use_known_spectrum: bool = field(init=True, default=False)
    
    def __post_init__(self):
        super().__post_init__()
        import ariel_load
        self.loaders = ariel_load.default_loaders()        

    def _check_constraints(self):
        assert(self.state>=0 and self.state<=1)
        assert len(self.loaders)==2
        for load in self.loaders:
            load.check_constraints()

    def train(self, train_data):
        train_data = copy.deepcopy(train_data)
        for t in train_data:            
            t.check_constraints()
        if self.state>=1:
            return        
        #if self.seed is None:
        #    self.seed = np.random.default_rng(seed=None).integers(0,1e6).item()    
        self._train(train_data)
        self.state = 1
        self.check_constraints()        

    def _train(self,train_data):
        pass
        # No training needed if not overridden

    #@profile_each_line
    def infer(self, test_data):
        assert self.state == 1
        test_data = copy.deepcopy(test_data)

        for t in test_data:
            if self.use_known_spectrum:
                assert not t.spectrum is None
            else:
                t.unload_spectrum()
            t.check_constraints()
        test_data_inferred = self._infer(test_data)
        assert([d.planet_id for d in test_data]==[d.planet_id for d in test_data_inferred])
        for t in test_data_inferred:
            t.check_constraints()
                
        return test_data_inferred

    def _infer(self, test_data):
        # Subclass must implement this OR _infer_single
        if self.run_in_parallel and not disable_any_parallel:  
            clear_gpu()
            dill_save(temp_dir + '/parallel.pickle', (self, sanity_checks_active, sanity_checks_without_errors, debugging_mode, profiling, n_threads))
            try:
                #from threadpoolctl import threadpool_limits
                #with threadpool_limits(limits=1):  # or 2 if each task is light
                    with multiprocess.Pool(recommend_n_workers()) as p:                   
                        result = list(tqdm(
                            p.imap(infer_internal_single_parallel, test_data),
                            total=len(test_data),
                            desc="Processing in parallel", smoothing = 0.05
                            ))
            except Exception as err:
                print('Planet ID', dill_load(temp_dir+'error.pickle')[1])
                raise
            
        else:
            result = []
            for d in tqdm(test_data, desc="Inferring", disable = len(test_data)<=1, smoothing = 0.05):     
                data=copy.deepcopy(d)
                orig_step = data.transits[0].loading_step
                from threadpoolctl import threadpool_limits
                from contextlib import nullcontext
                if n_threads==np.inf:
                    env = nullcontext()
                else:
                    env = threadpool_limits(limits=n_threads)                 
                with env:
                    data = self._infer_single(data)
                if len(test_data)>1:
                    data.load_to_step(orig_step,self.loaders)     
                result.append(data)                
        return result
    
def make_submission_dataframe(data, include_sigma=True):
    submission = pd.read_csv(data_dir + '/sample_submission.csv')
    submission = submission[0:0]    
    if not include_sigma:
        submission = submission.iloc[:, :284]    
    for i,d in enumerate(data):
        spec_clipped = copy.deepcopy(d.spectrum)
        spec_clipped[spec_clipped<=0]= 1e-9
        if include_sigma:
            submission.loc[i] = np.concatenate(([d.planet_id], spec_clipped, np.sqrt(np.diag(d.spectrum_cov))))
        else:
            submission.loc[i] = np.concatenate(([d.planet_id], spec_clipped))
    submission = submission.astype({'planet_id':'int64'})
    return submission

def data_to_mats(data, reference_data):
    y_true = np.stack([d.spectrum for d in reference_data])
    y_pred = np.stack([d.spectrum for d in data])
    cov_pred = np.stack([d.spectrum_cov for d in data])
    
    
    return cp.array(y_true),cp.array(y_pred),cp.array(cov_pred)

def mats_to_data(data, reference_data, mats):
    y_true,y_pred,cov_pred = mats
    for d, rd, yt, yp, covp in zip(data,reference_data,y_true,y_pred,cov_pred):
        
        rd.spectrum = yt.get()
        d.spectrum = yp.get()
        d.spectrum_cov = covp.get()
        
    
    if debugging_mode>=2:
        mats_test = data_to_mats(data, reference_data)
        for ii in range(3):
            assert(cp.all(mats[ii]==mats_test[ii]))
    

# def score_metric_fast(y_true, y_pred, cov_pred):
#     #y_true = np.stack([d.spectrum for d in reference_data])
#     #y_pred = np.stack([d.spectrum for d in data])
#     #sigma_pred = np.stack([np.sqrt(np.diag(d.spectrum_cov)) for d in data])
#     sigma_pred = cp.array([cp.sqrt(cp.diag(cov)) for cov in cov_pred]).get()
#     y_true = y_true.get()
#     y_pred = y_pred.get()
#     n_wavelengths = 283
    
#     sigma_true = np.append(
#         np.array(
#             [
#                 1e-6,
#             ]
#         ),
#         np.ones(n_wavelengths - 1) * 1e-5,
#     )
    
#     naive_mean, naive_sigma = cp.mean(y_true), cp.std(y_true)
    
#     GLL_pred = sp.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
#     GLL_true = sp.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
#     GLL_mean = sp.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

#     # normalise the score, right now it becomes a matrix instead of a scalar.
#     ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

#     fgs_weight=57.846
#     weights = np.append(np.array([fgs_weight]), np.ones(n_wavelengths - 1))
#     weights = weights * np.ones_like(ind_scores)
#     submit_score = np.average(ind_scores, weights=weights)
#     return float(submit_score) # clipping between 0 and 1 removed

#@profile_each_line
def score_metric_fast(y_true, y_pred, cov_pred, fgs_weight=57.846):
    """
    y_true:  (N, W) cupy.ndarray
    y_pred:  (N, W) cupy.ndarray
    cov_pred: (N, W, W) cupy.ndarray  or list/tuple of (W, W) CuPy arrays/sparse matrices
    returns: Python float (copied from GPU)
    """
    # Ensure GPU arrays
    sigma_pred = cp.sqrt(cp.diagonal(cov_pred, axis1=-2, axis2=-1))  # (N, W)

    N, W = y_true.shape

    # Fixed per-wavelength sigma for the "true" model
    sigma_true = cp.concatenate(
        (cp.array([1e-6], dtype=y_true.dtype),
         cp.ones(W - 1, dtype=y_true.dtype) * 1e-5)
    )  # shape (W,)

    # Baseline (naive) mean/sigma across all entries
    naive_mean = y_true.mean()
    naive_sigma = y_true.std()

    # Numerically safe normal logpdf on GPU
    eps = cp.finfo(y_true.dtype).tiny

    def norm_logpdf(x, loc, scale):
        s = cp.maximum(scale, eps)
        z = (x - loc) / s
        return -0.5 * (cp.log(2 * cp.pi) + 2 * cp.log(s) + z**2)

    # Broadcast shapes:
    # y_true, y_pred, sigma_pred -> (N, W)
    # sigma_true -> (W,) (broadcast to (N, W))
    # naive_* -> scalars (broadcast to (N, W))
    GLL_pred = norm_logpdf(y_true, y_pred, sigma_pred)
    GLL_true = norm_logpdf(y_true, y_true, sigma_true)
    GLL_mean = norm_logpdf(y_true, naive_mean, naive_sigma)

    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    # Per-wavelength weights (duplicate across rows)
    w = cp.concatenate(
        (cp.array([fgs_weight], dtype=y_true.dtype),
         cp.ones(W - 1, dtype=y_true.dtype))
    )  # (W,)
    Wmat = cp.broadcast_to(w, ind_scores.shape)  # (N, W)

    submit_score = cp.average(ind_scores, weights=Wmat)  # scalar on GPU
    return float(submit_score.get())
    

def score_metric(data,reference_data,print_results=True):
    solution = make_submission_dataframe(reference_data, include_sigma=False)
    submission = make_submission_dataframe(data)
    
    solution_np = solution.iloc[:,1:284].to_numpy()
    submission_np = submission.iloc[:,1:284].to_numpy()
    
    #rms_error = rms(solution_np-submission.iloc[:,1:284].to_numpy())
    rms_error_fgs = rms(solution_np[:,:1]-submission_np[:,:1])
    rms_fgs_per = np.abs(solution_np[:,:1]-submission_np[:,:1])
    rms_error_fgs_median = np.median(rms_fgs_per)
    
    
    rms_error_airs = rms(solution_np[:,1:]-submission_np[:,1:])
    rms_error_airs_per = np.sqrt(np.mean( (solution_np[:,1:]-submission_np[:,1:])**2, 1 ))
    #print(rms_error_airs_per.shape)
    rms_error_airs_median = np.median(rms_error_airs_per)
    
    diff_airs = solution_np[:,1:]-submission_np[:,1:]
    rms_error_airs_var = rms(diff_airs-np.mean(diff_airs,1)[:,None])
    rms_error_airs_var_per = np.sqrt(np.mean( (diff_airs-np.mean(diff_airs,1)[:,None])**2, 1 ))
    rms_error_airs_var_median = np.median(rms_error_airs_var_per)
    
    score = _score(solution, submission, 'planet_id', np.mean(solution_np), np.std(solution_np), fgs_weight=57.846)
    
    if print_results:        
        print(f"Score:           {score:.4f}")
        print(f"RMS error FGS:   {1e6*rms_error_fgs:.5f} ppm")
        print(f"mRMS error FGS:  {1e6*rms_error_fgs_median:.5f} ppm")
        print(f"RMS error AIRS:  {1e6*rms_error_airs:.5f} ppm")
        print(f"mRMS error AIRS: {1e6*rms_error_airs_median:.5f} ppm")
        print(f"RMS error AIRSv: {1e6*rms_error_airs_var:.5f} ppm")
        print(f"mRMS error AIRSv:{1e6*rms_error_airs_var_median:.5f} ppm")
        
    if debugging_mode>=1:
        assert(score - score_metric_fast(*data_to_mats(data,reference_data)) < 1e-9)
    
    return score,rms_error_fgs,rms_error_airs

def write_submission_csv(df):
    df = copy.deepcopy(df)
    df = df.set_index("planet_id")
    df.to_csv(csv_dir + '/submission.csv')
        

class ParticipantVisibleError(Exception):
    pass

def _score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 1,
) -> float:
    """
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
        fsg_sigma_true: (float) standard deviation from the FSG1 instrument for the test set.
        airs_sigma_true: (float) standard deviation from the AIRS instrument for the test set.
        fgs_weight: (float) relative weight of the fgs channel
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError('Wrong number of columns in the submission')

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    sigma_true = np.append(
        np.array(
            [
                fsg_sigma_true,
            ]
        ),
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    )
    y_true = solution.values

    GLL_pred = sp.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    GLL_true = sp.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    GLL_mean = sp.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true))

    # normalise the score, right now it becomes a matrix instead of a scalar.
    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(submit_score) # clipping between 0 and 1 removed