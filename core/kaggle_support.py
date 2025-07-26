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
from tqdm import tqdm


'''
Determine environment and globals
'''

if os.path.isdir('/mnt/d/ariel2/'):
    env = 'local'
else:
    env = 'kaggle'
print(env)

profiling = False
debugging_mode = 2
verbosity = 1

match env:
    case 'local':
        data_dir = '/mnt/d/ariel2/data/'
        temp_dir = '/mnt/d/ariel2/temp'             
        code_dir = '/mnt/d/ariel2/code/core/' 
    case 'kaggle':
        data_dir = 'XXX'
        temp_dir = '/temp/'             
        code_dir = 'XXX'         
os.makedirs(temp_dir, exist_ok=True)

# How many workers is optimal for parallel pool?
def recommend_n_workers():
    return torch.cuda.device_count()

n_cuda_devices = recommend_n_workers()
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
        # skip methods, but let descriptors through
        if callable(val) and not isinstance(val, property):
            continue
        print(f"{name} = {val}")

def remove_and_make_dir(path):
    try: shutil.rmtree(path)
    except: pass
    os.makedirs(path)

# Helper class - doesn't allow new properties after construction, and enforces property types. Partially written by ChatGPT.
@dataclass
class BaseClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)

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
    if env=='local':
        source=source.replace('/', '\\')
    subprocess.run('kaggle datasets version -p ' + source + ' -m ''Update''', shell=True)

def rms(array):
    return np.sqrt(np.mean(array**2))

def to_cpu(array):
    if isinstance(array, cp.array):
        return array.get()
    else:
        return array

def to_gpu(array):
    if isinstance(array, cp.array):
        return array
    else:
        return cp.array(array)
    
def sanity_check(f,to_check,name,code,limit):
    pass
sanity_checks_active = True

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
    Rs:  float = field(init=True, default=None) # Stellar radius in solar radii (R☉).
    Ms:  float = field(init=True, default=None) # Stellar mass in solar masses (M☉).
    Ts:  float = field(init=True, default=None) # Stellar effective temperature in Kelvin.
    Mp:  float = field(init=True, default=None) # Planetary mass in Earth masses (M⊕).
    e:   float = field(init=True, default=None) # Orbital eccentricity (dimensionless).
    P:   float = field(init=True, default=None) # Orbital period in days.
    sma: float = field(init=True, default=None) # Semi-major axis in stellar radii (Rs), showing the orbital distance relative to the stellar radii.
    i:   float = field(init=True, default=None) # Orbital inclination in degrees.
    
    spectrum: np.ndarray = field(init=True, default=None) # (Rp/Rs)^2, can be None or 1D array with lenght of wavelengths
    spectrum_cov: np.ndarray = field(init=True, default=None) # covariance matrix for uncertainty in spectrum, can be None or 2D array
    
    transits: list = field(init=True, default_factory = list) # transit observations for this planet
    
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
        self.Rs = row['Rs']
        self.Ms = row['Ms']
        self.Ts = row['Ts']
        self.Mp = row['Mp']
        self.e = row['e']
        self.P = row['P']
        self.sma = row['sma']
        self.i = row['i']
        
    def load_spectrum(self):
        assert self.is_train
        self.spectrum = train_labels[train_star_info['planet_id']==self.planet_id].to_numpy()
        assert self.spectrum.shape[0]==1    
        self.spectrum = self.spectrum[0,1:]
        self.spectrum_cov=None        
    
    def unload_spectrum(self):
        self.spectrum = None
        self.spectrum_cov = None
        
    def get_directory(self):
        if self.is_train:
            return data_dir + '/train/' + str(self.planet_id) + '/'
        else:
            return data_dir + '/test/' + str(self.planet_id) + '/'
        
@dataclass
class Transit(BaseClass):
    # Holds all relevant information for a single transit    
    planet: Planet = field(init=True, default=None) # linkback
    observation_number: int = field(init=True, default=None) # how manieth transit is this
    loading_step: int = field(init=True, default=0) # internal state
    # 0: unloaded
    # 1: raw parquet loaded
    # 2: pixel level corrections up to correlated double sampling done
    # 3: full-sensor corrections done
    # 4: wavelength and time binning done
    data: list = field(init=True, default_factory = lambda:[SensorData(is_FGS=True), SensorData(is_FGS=False)]) # 0: FGS, 1: AIRS
    
    def _check_constraints(self):
        assert len(self.data)==2
        assert self.data[0].is_FGS
        assert not self.data[1].is_FGS
        for d in self.data:
            assert(isinstance(d, SensorData))
            assert d.loading_step == self.loading_step
            d.check_constraints()
    
    def load_to_step(self, target_step, loaders):
        self.check_constraints()
        if target_step<self.loading_step:
            self.loading_step = 0
            self.data = Transit().data
            self.check_constraints()
        for loader in loaders:
            assert isinstance(loader, TransitLoader)
            loader.check_constraints()
        for ii in range(target_step - self.loading_step):
            for d,loader in zip(self.data, loaders):
                loader.progress_one_step(d, self.planet, self.observation_number)
            self.loading_step += 1
            self.check_constraints()
        assert self.loading_step == target_step
        
@dataclass
class SensorData(BaseClass):
    # Holds data for one sensor and one transit
    is_FGS: bool = field(init=True, default=None)
    loading_step: int = field(init=True, default=0) # internal state, see Transit
    
    data: cp.ndarray = field(init=True, default=None) 
    # 1st dimension: time
    # 2nd dimension: wavelength, but only if in step 4
    # further dimensions (0, 1, or 2): sensor coordinates
    times: cp.ndarray = field(init=True, default=None) # time associated with dimension 1 above, in seconds
    time_intervals: cp.ndarray = field(init=True, default=None) # time integration lengths, same size as above, in seconds
    wavelengths: cp.ndarray = field(init=True, default=None) # wavelengths associated with dimension 2 above, in um
        
    
    def _check_constraints(self):
        if self.loading_step==0:
            assert self.data is None
        elif self.loading_step==1:
            pass
        
class TransitLoader(BaseClass):
    # Manages loading transit data. User must fill in the classes below.
    
    load_raw_data: BaseClass = field(init=True, default=None) # must be callable
    apply_pixel_corrections: BaseClass = field(init=True, default=None) # must be callable
    apply_full_sensor_corrections: BaseClass = field(init=True, default=None) # must be callable
    apply_binning: BaseClass = field(init=True, default=None) # must be callable
    
    def progress_one_step(self, data, planet, observation_number):
        match data.loading_step:
            case 0:
                self.load_raw_data(data, planet, observation_number)
            case 1:
                self.apply_pixel_corrections(data, planet, observation_number)
            case 2:
                self.apply_full_sensor_corrections(data, planet, observation_number)
            case 3:
                self.apply_binning(data, planet, observation_number)
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
        
    files = glob.glob(data.get_directory()+'*')
    assert len(files)%4 == 0
    n_transits = len(files)//4
    for ii in range(n_transits):
        data.transits.append(Transit(planet=data))
        data.transits[-1].observation_number = ii
    
    data.check_constraints()
    
    return data

def load_all_train_data():
    return [load_data(ind,True) for ind in train_star_info['planet_id']]

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
        global disable_caching
        global cache_dir_read
        if model_parallel is None:
            model_parallel,disable_caching,cache_dir_read = dill_load(temp_dir+'parallel.pickle')
        t=time.time()
        if data.seismogram.data is None:
            data.seismogram.load_to_memory()
        return_data = model_parallel._infer_single(data)
        return_data.seismogram.unload()
        with portalocker.Lock(timing_filename, mode='a+', timeout=None, newline='') as csvfile:
            #csvfile.seek(0, os.SEEK_END)
            writer = csv.writer(csvfile)
            writer.writerow([data.cache_name(), data.family, time.time()-t])
        if model_parallel.write_cache and not return_data.do_not_cache and not disable_caching: # will be done later too, but in case we error out later...
            this_cache_dir = cache_dir_write+model_parallel.cache_name+'/'
            os.makedirs(this_cache_dir,exist_ok=True)
            dill_save(this_cache_dir+return_data.cache_name(), return_data.velocity_guess)
        return return_data
    except Exception as err:
        import traceback
        print(traceback.format_exc())     
        raise


@dataclass
class Model(BaseClass):
    # Loads one or more cryoET measuerements
    state: int = field(init=False, default=0) # 0: untrained, 1: trained    
    run_in_parallel: bool = field(init=False, default=False) 
    seed: object = field(init=True, default=None)  
    cache_name: str = field(init=True, default='')

    write_cache: bool = field(init=True, default=False)
    read_cache: bool = field(init=True, default=False)
    only_use_cached: bool = field(init=True, default=False)
    apply_offset: float = field(init=True, default=0.)
    round_results: bool = field(init=True, default=False)

    def _check_constraints(self):
        assert(self.state>=0 and self.state<=1)

    def train(self, train_data, validation_data):
        if self.state>=1:
            return
        if cache_only_mode:
            self.state=1
            return
        if self.seed is None:
            self.seed = np.random.default_rng(seed=None).integers(0,1e6).item()
        train_data = copy.deepcopy(train_data)
        validation_data = copy.deepcopy(validation_data)
        for d in train_data:
            d.unload()
        for d in validation_data:
            d.unload()
        self._train(train_data, validation_data)
        for d in train_data:
            d.unload()
        for d in validation_data:
            d.unload()
        self.state = 1
        self.check_constraints()        

    def _train(self,train_data, validation_data):
        pass
        # No training needed if not overridden

    #@profile_each_line
    def infer(self, test_data):
        assert self.state == 1
        test_data = copy.deepcopy(test_data)

        if self.read_cache and not disable_caching:
            this_cache_dir = cache_dir_read+self.cache_name+'/'
            files = set([os.path.basename(x) for x in glob.glob(this_cache_dir+'/*')])
            cached = []
            test_data_cached = []
            tt = copy.deepcopy(test_data)
            test_data = []
            for d in tqdm(tt, desc="Importing cache "+self.cache_name, disable=len(tt)<=1):
                if d.cache_name() in files:
                    cached.append(True)
                    test_data_cached.append(d)
                    test_data_cached[-1].velocity_guess = dill_load(this_cache_dir+d.cache_name())[0]
                else:
                    cached.append(False)
                    test_data.append(d)
       
        for t in test_data:
            if not t.velocity is None:
                t.velocity.unload()
        if self.only_use_cached:
            test_data_inferred = test_data
        else:
            if len(test_data)>0:
                test_data_inferred = self._infer(test_data)
            else:
                test_data_inferred = []

        if self.read_cache and not disable_caching:
            b_it = iter(test_data_cached)
            c_it = iter(test_data_inferred)        
            test_data = [
                next(b_it) if c else next(c_it)
                for c in cached
            ] 
        else:
            test_data = test_data_inferred

        for t in test_data:
            t.seismogram.unload()
            t.check_constraints()

        if self.write_cache and not disable_caching:
            this_cache_dir = cache_dir_write+self.cache_name+'/'
            os.makedirs(this_cache_dir,exist_ok=True)
            for d in test_data_inferred:
                if not d.do_not_cache:
                    dill_save(this_cache_dir+d.cache_name(), (d.velocity_guess, git_commit_id))

        for d in test_data:
            d.velocity_guess.data += self.apply_offset
            d.velocity_guess.min_vel += self.apply_offset
            if self.round_results:
                d.velocity_guess.data = np.round(d.velocity_guess.data)
                d.velocity_guess.min_vel = np.round(d.velocity_guess.min_vel)
                
        return test_data

    def _infer(self, test_data):
        # Subclass must implement this OR _infer_single
        if self.run_in_parallel:
            with open(timing_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["cache_name","family","time_taken"])
            for t in test_data:
                t.unload()
            claim_gpu('cupy')
            claim_gpu('pytorch')
            claim_gpu('')
            with multiprocess.Pool(recommend_n_workers()) as p:
                dill_save(temp_dir+'parallel.pickle', (self,disable_caching,cache_dir_read))
                #result = p.starmap(infer_internal_single_parallel, zip(test_data))            
                result = list(tqdm(
                    p.imap(infer_internal_single_parallel, test_data),
                    total=len(test_data),
                    desc="Processing in parallel "+self.cache_name, smoothing = 0.05
                    ))
        else:
            result = []
            for xx in tqdm(test_data, desc="Inferring "+self.cache_name, disable = len(test_data)<=1, smoothing = 0.05):     
                x = copy.deepcopy(xx)  
                if x.seismogram.data is None:
                    x.seismogram.load_to_memory()
                x = self._infer_single(x)
                x.seismogram.unload()       
                if self.write_cache and not x.do_not_cache and not disable_caching: # will be done later too, but in case we error out later...
                    this_cache_dir = cache_dir_write+self.cache_name+'/'
                    os.makedirs(this_cache_dir,exist_ok=True)
                    dill_save(this_cache_dir+x.cache_name(), x.velocity_guess)
                result.append(x)
        return result

@dataclass
class ChainedModel(Model):
    models: list = field(init=True, default_factory=list)

    def _train(self,train_data, validation_data):
        for m in self.models:
            m.train(train_data, validation_data)

    def _infer(self, data):
        for m in self.models:
            data = m.infer(data)
        return data

def score_metric(data, show_diagnostics=True):
    res_all = []
    res_per_family = dict()
    for d in data:
        d.velocity.load_to_memory()
        this_error = np.mean(np.abs(cp.asnumpy(d.velocity.data) - np.round(d.velocity_guess.data)))
        d.velocity.unload()
        res_all.append(this_error)
        if not d.family in res_per_family:
            res_per_family[d.family] = []
        res_per_family[d.family].append(this_error)

    score = np.mean(res_all)
    score_per_family = dict()
    score_per_family['family'] = res_per_family.keys()
    score_per_family['score'] = [np.mean(x) for x in res_per_family.values()]
    score_per_family = pd.DataFrame(score_per_family)

    if show_diagnostics:
        print(score_per_family)
        print('Combined: ', score)

    return score,score_per_family,res_all

# def write_submission_file(data, output_file = output_dir+'submission.csv'):
#     res = dict()
#     res['oid_ypos'] = []
#     x_vals = np.arange(1,70,2)
#     x_vals_names = [ 'x_'+str(x) for x in x_vals ]
#     for xn in x_vals_names:
#         res[xn] = []
#     for ii,d in enumerate(data):
#         if ii%100==0:print(ii)
#         name = os.path.basename(d.seismogram.filename[:-4])+'_y_'
#         data = np.round(d.velocity_guess.data).astype(int)
#         for y in np.arange(70):
#             res['oid_ypos'].append(name+str(y))
#             for x,xn in zip(x_vals, x_vals_names):
#                 res[xn].append(data[y,x])
#     print('x')
#     df = pd.DataFrame(res)
#     print('xx')
#     df.to_csv(output_file, index=False)
            
# def write_submission_file(data, output_file = output_dir+'submission.csv', obfuscate=0., do_round = False, do_range = True):
#     # precompute x‐positions and header
#     x_vals = np.arange(1, 70, 2)
#     x_names = [f"x_{x}" for x in x_vals]
#     header = ["oid_ypos"] + x_names

#     r=np.random.default_rng(seed=0)

#     with open(output_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

#         for ii, d in enumerate(data):

#             # your string prefix
#             base = os.path.basename(d.seismogram.filename)[:-4]
#             name_prefix = f"{base}_y_"

#             # grab and round your 70×70 numpy array
#             arr = d.velocity_guess.data.astype(np.float32) + (r.uniform(size=d.velocity_guess.data.shape)>0.5)*obfuscate
#             if do_round:
#                 arr = np.round(arr).astype(np.int32)
#             else:
#                 arr = arr.astype(np.float32)
#             if do_range:
#                 arr = np.clip(arr,1500,4500)

#             # slice out only the 35 columns you care about
#             sub = arr[:, x_vals]  # shape = (70, 35)

#             # stream each of the 70 rows
#             for y in range(sub.shape[0]):
#                 writer.writerow([f"{name_prefix}{y}"] + sub[y].tolist())