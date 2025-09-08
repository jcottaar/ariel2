'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module contains the basic Bayesian Inference framework, including the Gaussian Processes. It has no ARIEL competition specific content; that is in ariel_gp.py.

This module may look like a generic Gaussian Process toolbox, but be careful. Many features may only work when used in a specific way (i.e. what I ended up using in the competition) and may not work for general use cases. Also, be aware that the most typical use of GPs is interpolation, and that is not supported by this toolbox at all. Rather, it is focused on extracting individual model components from the posterior (most notably the transit depth).

It is only lightly commented; if you really want to understand this in depth you will likely not find it possible with what is provided.

A general feature is that many classes have an external and internal version of a function (the latter marked by '_internal'). The external one is meant to be used by external callers, and will call the internal one (typically implemented by a subclass).
'''

import numpy as np
import scipy as sp
import pandas as pd
import cupy as cp
import cupyx.scipy.sparse
import copy
import matplotlib.pyplot as plt
import kaggle_support as kgs # used for some support functions

# if kgs.running_on_kaggle:
#     # Load sksparse, which is not installed by default. This is the best I managed after half a day of messing with pip, conda, and apt-get...
#     import subprocess
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/sksparse.zip -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libcholmod.so.3 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libamd.so.2 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libcamd.so.2 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libcolamd.so.2 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libccolamd.so.2 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so.5 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("unzip -n /kaggle/usr/lib/install_skparse_pip/lib.zip usr/lib/x86_64-linux-gnu/libmetis.so.5 -d /", shell=True, stdout=subprocess.DEVNULL)
#     subprocess.run("export CHOLMOD_USE_GPU=1", shell=True, stdout=subprocess.DEVNULL)
from sksparse.cholmod import cholesky

# Define sparse matrix class we use throughout
sparse_matrix = sp.sparse.csc_matrix
sparse_matrix_gpu = cupyx.scipy.sparse.csc_matrix
sparse_matrix_str = 'csc'

def check_matrix(A):
    assert str(A.format) == sparse_matrix_str
    assert str(A.dtype) == 'float64'
    assert str(A.indptr.dtype) in ['int32','int64']
    assert str(A.indices.dtype) in ['int32','int64']

# Try to get sksparse to use GPU, but don't think it works
import os
os.environ["CHOLMOD_USE_GPU"] = "1"

class Observable(kgs.BaseClass):
    # Holds measured data
    df = 0 # features (as pandas DataFrame)
    labels = np.zeros((0,0)) # labels, rows are features, columns are samples (in input there will be only 1 column)
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()
    @property
    def number_of_observations(self): # number of measurements
        return self.labels.shape[0]
    @property
    def number_of_instances(self): # number of samples
        return self.labels.shape[1]

    def select_observations(self, which):
        self.df = self.df[which]
        self.labels = self.labels[which,:]
    def _check_constraints(self):
        assert isinstance(self.df, pd.DataFrame)
        assert isinstance(self.labels, np.ndarray)
        assert len(self.df) == self.labels.shape[0]
        assert len(self.labels.shape) == 2
        super()._check_constraints()

class PriorMatrices(kgs.BaseClass):
    # Holds the properties of the model in matrix form
    # Distribution of parameters: Gaussian with mean prior_mean and K prior_precision_matrix^-1
    # Relationship to observations (linearized around current parameter values): obs = design_matrix*par + observable_offset
    number_of_observations = 0
    number_of_parameters = 0
    prior_mean = np.zeros((0))
    observable_offset = np.zeros((0))
    prior_precision_matrix = sparse_matrix((0,0))
    design_matrix = sparse_matrix((0,0))
    noise_parameter_indices = np.zeros((0))
    def _check_constraints(self):
        assert isinstance(self.number_of_observations, int)
        assert isinstance(self.number_of_parameters, int)
        assert isinstance(self.prior_mean, np.ndarray)
        assert isinstance(self.observable_offset, np.ndarray)
        #assert isinstance(self.prior_precision_matrix, sparse_matrix)
        #assert isinstance(self.design_matrix, sparse_matrix)
        check_matrix(self.prior_precision_matrix)
        check_matrix(self.design_matrix)
        assert isinstance(self.noise_parameter_indices, np.ndarray)
        assert self.prior_mean.shape==(self.number_of_parameters,)
        assert self.observable_offset.shape==(self.number_of_observations,)
        assert self.prior_precision_matrix.shape==(self.number_of_parameters,self.number_of_parameters)
        assert self.design_matrix.shape==(self.number_of_observations,self.number_of_parameters)
        assert self.noise_parameter_indices.shape == (0,) or self.noise_parameter_indices.shape == (self.number_of_observations,)
        super()._check_constraints()

class Model(kgs.BaseClass):
    # The core class; this one is abstract, and subclasses will define actual behavior. Typical flow:
    # - initialize with initialize(), informing the model about the observable
    # - find the distribution in matrix form with get_prior_matrices()
    # - find posterior mean and samples in vector form
    # - put these into the model with set_parameters()
    number_of_observations = 0
    number_of_instances = 0
    number_of_parameters = 0
    number_of_hyperparameters = 0
    expected_parameter_scale = 1. # only used for sanity checks
    expected_observation_scale = 1. # only used for sanity checks
    c1 = 1e-5 # only used for sanity checks
    c2 = 1e-8 # only used for sanity checks

    scaling_factor = 1. # Rows and columns of prior distribution matrix will be multiplied by this
    update_scaling = False # Should we use scaling_factor as a hyperparameter to tune?
    force_cache_observation_relationship = False # Cache even if we're not linear?

    initialized = False # keep track of state, run initialize() to set to True

    # caches to avoid recalculating matrices when not needed
    cached_prior_distribution = None
    cached_observation_relationship = None
    
    def _check_constraints(self):
        assert isinstance(self.number_of_observations, int)
        assert isinstance(self.number_of_instances, int)
        assert isinstance(self.number_of_parameters, int)
        assert isinstance(self.number_of_hyperparameters, int)
        assert isinstance(self.expected_parameter_scale, float)
        assert isinstance(self.expected_observation_scale, float)
        super()._check_constraints()
        
    def initialize(self, obs, number_of_instances=1):
        # Initialize our parameters based on an observable (typically just setting the appropriate amount of zeroes); labels in the observable are not used
        assert isinstance(obs, Observable)        
        obs.check_constraints()
        assert not self.initialized
        self.initialized = True       
        
        self.initialize_internal(obs, number_of_instances)
        self.number_of_observations = obs.number_of_observations
        self.number_of_instances = number_of_instances        

        if self.update_scaling:
            self.number_of_hyperparameters = 1
        
        self.check_constraints()
    
    def set_parameters(self, to_what):
        # Set parameters to given values
        assert isinstance(to_what, np.ndarray)
        assert to_what.shape[0] == self.number_of_parameters
        self.number_of_instances = to_what.shape[1]
        self.set_parameters_internal(to_what)
        self.check_constraints()
        if not self.is_linear():
            self.cached_observation_relationship = None
        if kgs.debugging_mode>=2:
            assert np.all(np.isclose(to_what, self.get_parameters()))

    def get_parameters(self):
        # Return stored parameters
        result = self.get_parameters_internal()
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.number_of_parameters, self.number_of_instances)
        return result
 
    def get_prior_matrices(self, obs, get_prior_distribution=True, get_observation_relationship=True):
        # Get the distribution in matrix form
        assert self.number_of_instances == 1
        assert obs.number_of_observations == self.number_of_observations

        if get_prior_distribution:
            if not self.cached_prior_distribution is None:
                prior_matrices = copy.deepcopy(self.cached_prior_distribution)
            else:
                prior_matrices = self.get_prior_distribution_internal(obs)
                self.cached_prior_distribution = copy.deepcopy(prior_matrices)
        else:
            prior_matrices = PriorMatrices()
            prior_matrices.number_of_parameters = self.number_of_parameters
            prior_matrices.prior_mean = np.zeros((prior_matrices.number_of_parameters))
            prior_matrices.prior_precision_matrix = sparse_matrix((prior_matrices.number_of_parameters,prior_matrices.number_of_parameters))
        if get_observation_relationship:
            if not self.cached_observation_relationship is None:
                prior_matrices_obs = copy.deepcopy(self.cached_observation_relationship)
            else:
                prior_matrices_obs = self.get_observation_relationship_internal(obs)
                if self.is_linear() or self.force_cache_observation_relationship:
                    self.cached_observation_relationship = copy.deepcopy(prior_matrices_obs)
            prior_matrices.observable_offset = prior_matrices_obs.observable_offset
            prior_matrices.design_matrix = prior_matrices_obs.design_matrix
            prior_matrices.number_of_observations = prior_matrices_obs.number_of_observations
        else:
            prior_matrices.design_matrix = sparse_matrix((self.number_of_observations, self.number_of_parameters))
            prior_matrices.observable_offset = np.zeros((self.number_of_observations))            
            prior_matrices.number_of_observations = self.number_of_observations
        if not self.scaling_factor==1.:
            assert np.all(prior_matrices.prior_mean)==0 # todo
            diag_scale = sp.sparse.diags(np.zeros((self.number_of_parameters)) + 1/self.scaling_factor)
            prior_matrices.prior_precision_matrix = sparse_matrix(diag_scale @ (prior_matrices.prior_precision_matrix @ diag_scale))

        # Ensure symmetry
        prior_matrices.prior_precision_matrix = (prior_matrices.prior_precision_matrix + prior_matrices.prior_precision_matrix.T)/2
        
        assert prior_matrices.number_of_observations == self.number_of_observations
        assert prior_matrices.number_of_parameters == self.number_of_parameters
        prior_matrices.check_constraints()
        if kgs.debugging_mode>=2 and get_observation_relationship:
            # Check design matrix against get_prediction()
            pred1 = self.get_prediction_internal(obs)
            offset = np.random.default_rng(seed=0).normal(scale=1e-6*self.expected_parameter_scale,size=(self.number_of_parameters,1))
            model_test = copy.deepcopy(self)
            model_test.set_parameters(model_test.get_parameters_internal() + offset)
            pred2 = model_test.get_prediction_internal(obs)
            assert np.all(np.abs(pred1.T - (prior_matrices.observable_offset + (prior_matrices.design_matrix@self.get_parameters_internal()).T)) <=1e-10*np.abs(pred1.T)+1e-10*self.expected_observation_scale)
            diff_actual = pred2-pred1
            diff_pred =  prior_matrices.design_matrix@offset   
            #import ariel_gp
            #if isinstance(self, ariel_gp.TransitModel):
            #    print(kgs.rms(diff_actual - diff_pred), kgs.rms(diff_actual+diff_pred))
            assert np.all(np.abs(diff_actual - diff_pred)<=self.c1*np.abs(diff_actual+diff_pred)+self.c2*self.expected_observation_scale)
        
        return prior_matrices

    def get_prediction(self, obs):
        # Predict label values from the parameters
        assert self.number_of_observations == obs.number_of_observations
        assert self.number_of_instances > 0
        labels = self.get_prediction_internal(obs)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (obs.number_of_observations, self.number_of_instances)
        
        return labels

    def is_linear(self):
        # Are our predictions a linear function of the parameters?
        return True # Default, subclass needs to override this if it doesn't hold

    def set_hyperparameters(self, to_what):
        # Set our hyperparameters, which determine the distribution of the parameters
        assert isinstance(to_what, np.ndarray)
        assert to_what.shape == (self.number_of_hyperparameters,)
        old_hyper = self.get_hyperparameters()
        if self.update_scaling:
            # Special case: scaling_factor is our hyperparameter. This overrides whatever opinion the subclass may have.
            old_scaling = self.scaling_factor
            self.scaling_factor = 1/np.sqrt(to_what)[0]
        else:
            self.set_hyperparameters_internal(to_what)
        if kgs.debugging_mode>=2:
            assert np.all(np.abs(self.get_hyperparameters()-to_what) <= 1e-10*np.abs(self.get_hyperparameters()+to_what))        
        if not np.all(old_hyper == to_what):
            self.cached_prior_distribution = None

    def set_hyperparameters_internal(self, to_what):
        pass # Default: no hyperparameters

    def get_hyperparameters(self):
        if self.update_scaling:
            # Special case: scaling_factor is our hyperparameter. This overrides whatever opinion the subclass may have.
            return np.reshape(1/self.scaling_factor**2, 1)
        else:
            return self.get_hyperparameters_internal()

    def get_hyperparameters_internal(self):
        return np.zeros((0))

    def update_hyperparameters(self):
        clear_cache1 = self.update_scaling_factor()
        clear_cache2 = self.update_hyperparameters_internal()
        
        # Scaling factor doesn't need a cache clear
        if clear_cache2:
            self.cached_prior_distribution = None

        # Parent needs to redo prior matrices if anything was changed
        return (clear_cache1 or clear_cache2)

    def update_hyperparameters_internal(self):
        return False

    def update_scaling_factor(self):
        if self.update_scaling:
            # Adapt scaling_factor to maximize the log likelihood
            X = self.get_parameters()
            P = self.cached_prior_distribution.prior_precision_matrix  # we're just assuming this is cached...
            old_scaling = self.scaling_factor
            self.scaling_factor = np.sqrt(np.sum(X*(P@X))/(X.shape[0]*X.shape[1]))
            return True
        else:
            return False

    def get_partial_prior_precision_matrices(self,obs):
        # Get the derivative of the precision matrix to our hyperparameters
        if self.update_scaling:
            prior_matrices = self.get_prior_matrices(obs, get_observation_relationship=False)
            prior_matrices.check_constraints()
            matrices = [prior_matrices.prior_precision_matrix] 
        else:
            matrices = self.get_partial_prior_precision_matrices_internal(obs)
        assert len(matrices) == self.number_of_hyperparameters
        for m in matrices:
            assert m.shape == (self.number_of_parameters, self.number_of_parameters)
            if kgs.debugging_mode>=1:
                check_matrix(m)
        return matrices

    def get_partial_prior_precision_matrices_internal(self,obs):
        # Default: no hyperparameters, so no derivatives
        return []

    def clear_all_caches(self):
        # Clear caches
        self.cached_prior_distribution = None    
        self.cached_observation_relationship = None


class FixedShape(Model):
    # A model with no parameters, just returning some fixed shape. Still abstract; subclass must define the actual shape.
    def initialize_internal(self,obs,number_of_instances):
        pass

    def set_parameters_internal(self, to_what):
        pass

    def get_parameters_internal(self):
        return np.zeros((0,self.number_of_instances))

    def get_observation_relationship_internal(self,obs):
        prior_matrices = PriorMatrices()
        prior_matrices.observable_offset = self.get_prediction_internal(obs)[:,0] # internal to avoid offset adding
        prior_matrices.design_matrix = sparse_matrix( np.zeros((self.number_of_observations, 0)) )
        prior_matrices.number_of_observations = self.number_of_observations
        return prior_matrices

    def get_prior_distribution_internal(self,obs):
        return PriorMatrices()

    def is_linear(self):
        return True

class FixedValue(FixedShape):
    # Always equal to "offset" in all measurement points. Finally a class that is not abstract!
    offset = 0.
    
    def get_prediction_internal(self,obs):
        return (np.zeros((obs.number_of_observations, self.number_of_instances)).T + self.offset).T

class Passthrough(Model):
    # Behaves exactly as its _model property, i.e. all calls get passed to it. Typically subclasses of this class will make it do something more interesting.
    _model = []
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self,value):
        self._model = value
    
    def _check_constraints(self):
        assert isinstance(self.model, Model)
        assert (self.number_of_instances==self.model.number_of_instances)
        self.model.check_constraints()
        super()._check_constraints()

    def initialize_internal(self,obs,number_of_instances):
        self.model.comment = self.comment
        self.model.initialize(obs, number_of_instances)
        self.number_of_parameters = self.model.number_of_parameters
        self.number_of_hyperparameters = self.model.number_of_hyperparameters

    def set_parameters_internal(self, to_what):
        self.model.set_parameters(to_what)

    def get_parameters_internal(self):
        return self.model.get_parameters()

    def set_hyperparameters_internal(self, to_what):
        self.model.set_hyperparameters(to_what)

    def get_hyperparameters_internal(self):
        return self.model.get_hyperparameters()

    def get_observation_relationship_internal(self,obs):
        return self.model.get_prior_matrices(obs, get_prior_distribution=False)

    def get_prior_distribution_internal(self,obs):
        return self.model.get_prior_matrices(obs, get_observation_relationship=False) 
    
    def get_prediction_internal(self,obs):
        return self.model.get_prediction(obs)

    def is_linear(self):
        return self.model.is_linear()

    def update_hyperparameters_internal(self):
        return self.model.update_hyperparameters()

    def get_partial_prior_precision_matrices_internal(self,obs):
        return self.model.get_partial_prior_precision_matrices(obs)

    def clear_all_caches(self):
        self.model.clear_all_caches()
        super().clear_all_caches()

class ParameterScaler(Passthrough):
    # Ensure internal parameters for a class = (scaler * external_parameters)
    # Used to make all parameters ~1, which improves numerics

    scaler = 1.

    def _check_constraints(self):
        assert isinstance(self.scaler, float)

    def initialize_internal(self,obs,number_of_instances):
        super().initialize_internal(obs, number_of_instances)
        if self.model.expected_parameter_scale == 1.:
            self.model.expected_parameter_scale = self.scaler

    def set_parameters_internal(self, to_what):
        self.model.set_parameters(to_what*self.scaler)

    def get_parameters_internal(self):
        return self.model.get_parameters()/self.scaler

    def get_prior_distribution_internal(self,obs):
        prior_matrices = self.model.get_prior_matrices(obs, get_observation_relationship=False)
        prior_matrices.prior_mean = prior_matrices.prior_mean/self.scaler
        prior_matrices.prior_precision_matrix = prior_matrices.prior_precision_matrix*(self.scaler**2)
        return prior_matrices

    def get_observation_relationship_internal(self,obs):
        prior_matrices = self.model.get_prior_matrices(obs, get_prior_distribution=False)
        prior_matrices.design_matrix = prior_matrices.design_matrix*self.scaler
        return prior_matrices        

    def get_partial_prior_precision_matrices_internal(self,obs):
        matrices =  self.model.get_partial_prior_precision_matrices(obs)
        for i in range(len(matrices)):
            matrices[i] = matrices[i]*self.scaler**2
        return matrices



class Sparse2D(Passthrough):
    # KISS-GP implementation. The internal model is evaluated on a sparse grid, and the behavior on the full grid is determined by linear interpolation.
    h = None # 2-element array specifying grid resolution
    features = None # Which 2 features in the observable do we depend on?
    transform_matrix = None # Linear interpolation matrix (set internally)

    def __init__(self):
        grid_size = np.array([1,1])

    def _check_constraints(self):
        assert self.h.shape == (2,)
        assert len(self.features)==2
        super()._check_constraints()

    def alter_obs(self, obs):
        # Make the observable to pass to the internal mdoel
        obs_altered = Observable()
        x = obs.df[self.features[0]]
        y = obs.df[self.features[1]]
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        xt = np.linspace(min_x-self.h[0], max_x+self.h[0], int(np.round((max_x-min_x + self.h[0])/self.h[0])))
        yt = np.linspace(min_y-self.h[1], max_y+self.h[1], int(np.round((max_y-min_y + self.h[1])/self.h[1])))
        yy,xx = np.meshgrid(yt,xt)
        xx = np.reshape(xx,-1)
        yy = np.reshape(yy,-1)
        obs_altered.df['x'] = xx
        obs_altered.df['y'] = yy
        obs_altered.labels = np.tile(np.zeros(xx.shape) , (obs.number_of_instances,1)).T
        return obs_altered,xt,yt,x,y

    def initialize_internal(self, obs, number_of_instances):
        # Make grid
        obs_altered,xt,yt,x,y = self.alter_obs(obs)
        hx = xt[1]-xt[0]
        hy = yt[1]-yt[0]
        super().initialize_internal(obs_altered, number_of_instances)

        # This is as good a time as any to prepare our transformation matrix...
        ind_x_left = np.floor( (x-xt[0])/hx ).astype(int)
        scale_x = 1-(x-xt[ind_x_left])/hx
        assert np.all(np.logical_and(ind_x_left>=0, ind_x_left<=len(xt)-2))
        ind_y_top = np.floor ( (y-yt[0])/hy ).astype(int)
        scale_y = 1-(y-yt[ind_y_top])/hy
        assert np.all(np.logical_and(ind_y_top>=0, ind_y_top<=len(yt)-2))
        ind_top_left = ind_y_top + len(yt)*ind_x_left

        rows = np.arange(obs.number_of_observations) # repeated 4x        
        # top left
        cols1 = ind_top_left
        vals1 = scale_x * scale_y
        # top right
        cols2 = ind_top_left + len(yt)
        vals2 = (1-scale_x) * scale_y
        # bottom left
        cols3 = ind_top_left + 1
        vals3 = scale_x * (1-scale_y)
        # bottom right
        cols4 = ind_top_left + len(yt) + 1
        vals4 = (1-scale_x) * (1-scale_y)
 
        rows = np.concatenate((rows,rows,rows,rows))
        cols = np.concatenate((cols1,cols2,cols3,cols4))
        vals = np.concatenate((vals1,vals2,vals3,vals4))

        self.transform_matrix = sparse_matrix((vals, (rows.astype(int),cols.astype(int))), shape = (obs.number_of_observations,obs_altered.number_of_observations))
 
    def get_prior_distribution_internal(self,obs):
        return self.model.get_prior_matrices(self.alter_obs(obs)[0], get_observation_relationship=False) 
        
    def get_observation_relationship_internal(self,obs):
        prior_matrices = self.model.get_prior_matrices(self.alter_obs(obs)[0], get_prior_distribution=False) 
        prior_matrices.design_matrix = self.transform_matrix @ prior_matrices.design_matrix
        prior_matrices.observable_offset = self.transform_matrix @ prior_matrices.observable_offset
        prior_matrices.number_of_observations = self.number_of_observations
        return prior_matrices

    def get_prediction_internal(self,obs):
        return self.transform_matrix @ self.model.get_prediction(self.alter_obs(obs)[0])


class Compound(Model):
    # Combines multiple models; their predictions are added or multiplied
    mode = 'sum' # 'sum' or 'product'
    offset = 0. # offset added to the sum or product

    _models = 0 # which models to add/multiply
    @property
    def models(self):
        return self._models
    @models.setter
    def models(self,value):
        self._models = value

    def __init__(self):
        self._models = []
        super().__init__()
    
    def _check_constraints(self):
        assert isinstance(self.offset, float)
        assert isinstance(self.models, list)
        assert len(self.models)>0
        assert self.number_of_parameters == np.sum([x.number_of_parameters for x in self.models])
        if not self.update_scaling:
            assert self.number_of_hyperparameters == np.sum([x.number_of_hyperparameters for x in self.models])
        else:
            assert self.number_of_hyperparameters == 1
        assert self.mode == 'sum' or self.mode == 'product'
        assert np.all(self.number_of_observations == np.array([x.number_of_observations for x in self.models]))
        assert np.all(self.number_of_instances == np.array([x.number_of_instances for x in self.models]))
        for m in self.models:
            assert isinstance(m, Model)
            if kgs.debugging_mode>=2:
                m.check_constraints()
        super()._check_constraints()

    def initialize_internal(self, obs, number_of_instances):
        param = 0
        hparam = 0
        for m in self.models:
            m.initialize(obs, number_of_instances=number_of_instances)
            param = param+m.number_of_parameters
            hparam = hparam+m.number_of_hyperparameters
        self.number_of_parameters = param
        self.number_of_hyperparameters = hparam

    def set_parameters_internal(self, to_what):
        cur_pos = 0
        for m in self.models:
            next_pos = cur_pos+m.number_of_parameters
            m.set_parameters(to_what[cur_pos:next_pos,:])
            cur_pos = next_pos
        assert next_pos == self.number_of_parameters

    def get_parameters_internal(self):
        res_per_model = [m.get_parameters() for m in self.models]
        return np.concatenate(res_per_model, axis=0)

    def set_hyperparameters_internal(self, to_what):
        cur_pos = 0
        for m in self.models:
            next_pos = cur_pos+m.number_of_hyperparameters
            m.set_hyperparameters(to_what[cur_pos:next_pos])
            cur_pos = next_pos
        assert next_pos == self.number_of_hyperparameters

    def get_hyperparameters_internal(self):
        res_per_model = [m.get_hyperparameters() for m in self.models]
        return np.concatenate(res_per_model)

    def get_observation_relationship_internal(self,obs):
        pm = [m.get_prior_matrices(obs, get_prior_distribution = False) for m in self.models] # we get both -> rely on caching for efficiency
        prior_matrices = PriorMatrices()
        
        prior_matrices.number_of_observations = pm[0].number_of_observations
        assert np.all([p.number_of_observations==prior_matrices.number_of_observations for p in pm])

        if self.mode == 'sum':
            prior_matrices.observable_offset = np.sum([p.observable_offset for p in pm], axis=0)
            prior_matrices.design_matrix = concatenate_sparse_csc_matrices([p.design_matrix for p in pm])
        else:
            # product
            pred_per_model = [m.get_prediction(obs) for m in self.models]            
            for p in pred_per_model:
                p[p==0] = 1e-50
            assert not np.any(np.array(pred_per_model)==0)
            base_val = np.product(pred_per_model,axis=0)
            design_matrix_parts = [None for m in self.models]
            for i in range(len(self.models)):
                other_val = base_val/pred_per_model[i]
                design_matrix_parts[i] = sparse_matrix ( sp.sparse.diags(other_val[:,0]) @ pm[i].design_matrix ) 
            prior_matrices.design_matrix = concatenate_sparse_csc_matrices(design_matrix_parts)
            prior_matrices.observable_offset = base_val - prior_matrices.design_matrix@self.get_parameters()                
            prior_matrices.observable_offset = prior_matrices.observable_offset[:,0]

        prior_matrices.observable_offset = prior_matrices.observable_offset + self.offset

        return prior_matrices

    def get_prior_distribution_internal(self,obs):
        pm = [m.get_prior_matrices(obs, get_observation_relationship=False) for m in self.models]
        prior_matrices = PriorMatrices()
        
        prior_matrices.number_of_observations = pm[0].number_of_observations
        assert np.all([p.number_of_observations==prior_matrices.number_of_observations for p in pm])
      
        prior_matrices.number_of_parameters = sum([p.number_of_parameters for p in pm])
        
        prior_matrices.prior_mean = np.concatenate([p.prior_mean for p in pm])
        prior_matrices.prior_precision_matrix = sp.sparse.block_diag([p.prior_precision_matrix for p in pm], format=sparse_matrix_str)
        prior_matrices.noise_parameter_indices = np.zeros((0))
        is_noise = [len(p.noise_parameter_indices)>0 for p in pm]
        if sum(is_noise)==1:
            ind = np.argmax(is_noise)
            nparams_lower = sum([p.number_of_parameters for p in pm[0:ind]])
            prior_matrices.noise_parameter_indices = pm[ind].noise_parameter_indices + nparams_lower
        else:
            assert sum(is_noise)==0

        return prior_matrices

    def get_prediction_internal(self, obs):
        res = [m.get_prediction(obs) for m in self.models]
        if self.mode=='sum':
            return np.sum(res,axis=0) + self.offset
        else:
            return np.product(res,axis=0) + self.offset
 
    def is_linear(self):
        if self.mode=='sum':
            return np.all([m.is_linear() for m in self.models])
        else:
            return False

    def update_hyperparameters_internal(self):
        clear_cache = False
        for m in self.models:
            t=m.update_hyperparameters()
            if t:
                clear_cache = True
        return clear_cache

    def get_partial_prior_precision_matrices_internal(self,obs):
        matrices = []
        cur_pos = 0
        for m in self.models:
            next_pos = cur_pos + m.number_of_parameters
            this_matrices = m.get_partial_prior_precision_matrices(obs)
            for i in range(len(this_matrices)):
                old_indptr = this_matrices[i].indptr
                this_matrices[i].resize((self.number_of_parameters, self.number_of_parameters))
                this_matrices[i].indices = this_matrices[i].indices+int(cur_pos)
                this_matrices[i].indptr = np.zeros(len(this_matrices[i].indptr), dtype=this_matrices[i].indptr.dtype)
                this_matrices[i].indptr[cur_pos:next_pos+1] = old_indptr
                matrices.append(this_matrices[i])
                #raise('stop')
                #full_mat = sparse_matrix((self.number_of_parameters, self.number_of_parameters))
                #full_mat[cur_pos:next_pos, cur_pos:next_pos] = mat
                #matrices.append(full_mat)
            cur_pos = next_pos
        assert cur_pos == self.number_of_parameters
        return matrices

    def clear_all_caches(self):
        for m in self.models:
            m.clear_all_caches()
        super().clear_all_caches()

class CompoundNamed(Compound):
    # Allows accessing the models in Compound using a dict, making it easier to keep track of what is what in our models
    model_names = []
    def _check_constraints(self):
        assert isinstance(self.model_names, list)
        assert len(self.model_names) == len (self.models)        
        for s in self.model_names:
            assert isinstance(s, str)
        super()._check_constraints()

    @property
    def m(self):
        d = dict()
        for i in range(len(self.models)):
            d[self.model_names[i]] = self.models[i]
        return d

    @m.setter
    def m(self, d):
        self.model_names = list(d.keys())
        self.models = [d[x] for x in self.model_names]

    def initialize_internal(self, obs, number_of_instances):
        for i in range(len(self.models)):
            self.models[i].comment = self.comment + '.' + self.model_names[i]
        super().initialize_internal(obs, number_of_instances)

    

class ValueHolder(Model):
    # Abstract class that manages models that store their parameters explicitly.
    parameters = np.zeros((0,0)) # rows: parameteres, columns: samples
    def _check_constraints(self):
        assert isinstance(self.parameters, np.ndarray)
        assert self.parameters.shape == (self.number_of_parameters, self.number_of_instances)
        super()._check_constraints()

    def initialize_internal(self, obs, number_of_instances):
        self.parameters = np.zeros((self.get_number_of_parameters_internal(obs), number_of_instances))
        self.number_of_parameters = self.parameters.shape[0]

    def set_parameters_internal(self, to_what):
        self.parameters = to_what

    def get_parameters_internal(self):
        return self.parameters       

class FeatureSelector(ValueHolder):
    # Abstract class that manages models that depend only on a subset of features. If these features have the same value for more than one measurement, all these measurements will have the same prediction.
    features = []
    check_for_uniqueness=True

    # cached stuff
    pred=None
    mat=None
    perm=None
    def _check_constraints(self):
        assert isinstance(self.features, list)
        for f in self.features:
            assert isinstance(f, str)
        super()._check_constraints()

    def initialize_internal(self, obs, number_of_instances):        
        self.perm, self.mat = self.convert_features_to_nd_array_internal(obs)        
        self.initialize_from_mat_internal(self.mat)
        super().initialize_internal(obs, number_of_instances)
        

    def initialize_from_mat_internal(self, mat):
        pass

    def convert_features_to_nd_array_internal(self, obs):
        if obs.number_of_observations==0:
            unique_vals = np.zeros((0))
            perm = sparse_matrix((0,0))
            return perm, unique_vals
        elif not self.check_for_uniqueness:
            return sparse_matrix(sp.sparse.eye(obs.number_of_observations)), obs.df[self.features].to_numpy()
        else:
            base_mat = obs.df[self.features].to_numpy()
            unique_vals,inds = np.unique(base_mat,axis=0,return_inverse=True)
            n_obs = obs.number_of_observations
            perm = sparse_matrix( (np.ones(n_obs), (np.arange(n_obs), inds)) )
            return perm, unique_vals

    def get_number_of_parameters_internal(self, obs):
        return self.get_number_of_parameters_from_mat_internal(self.mat)

    def get_observation_relationship_internal(self,obs):       
        prior_matrices = self.get_observable_relationship_from_mat_internal(self.mat)
        prior_matrices.design_matrix = self.perm@prior_matrices.design_matrix
        prior_matrices.observable_offset = self.perm@prior_matrices.observable_offset
        prior_matrices.number_of_observations = obs.number_of_observations
        return prior_matrices

    def get_prior_distribution_internal(self,obs):
        return self.get_prior_distribution_from_mat_internal(self.mat)

    def get_prediction_internal(self, obs):
        return self.perm@self.get_prediction_from_mat_internal(self.mat)

    def update_hyperparameters_internal(self):
        return self.update_hyperparameters_from_mat_internal(self.mat)

    def update_hyperparameters_from_mat_internal(self,mat):
        return False

class FixedBasis(FeatureSelector):
    # Manages a fixed set of basis function. Prior covariance matrix is diagonal. 
    basis_functions = 0 # as matrix, # rows = number of observations, # cols = number of basis functions
    regularization_variance = 0 # variance values per basis function

    def _check_constraints(self):
        assert isinstance(self.basis_functions, np.ndarray)
        assert isinstance(self.regularization_variance, np.ndarray)
        assert len(self.basis_functions.shape)==2
        assert self.regularization_variance.shape == (self.basis_functions.shape[1],)
        super()._check_constraints()

    def get_number_of_parameters_from_mat_internal(self, mat):
        return self.basis_functions.shape[1]

    def get_observable_relationship_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        assert self.basis_functions.shape[0]==mat.shape[0]
        prior_matrices.number_of_observations = self.basis_functions.shape[0]    
        prior_matrices.observable_offset = np.zeros((mat.shape[0]))
        prior_matrices.design_matrix = sparse_matrix (self.basis_functions)
        return prior_matrices

    def get_prior_distribution_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()        
        prior_matrices.number_of_parameters = self.basis_functions.shape[1]
        prior_matrices.prior_mean = np.zeros((self.basis_functions.shape[1]))
        prior_matrices.prior_precision_matrix = sparse_matrix (np.diag(1/self.regularization_variance))
        return prior_matrices

    def get_prediction_from_mat_internal(self, mat):
        return self.basis_functions@self.parameters


class Uncorrelated(FeatureSelector):
    # Uncorrelated Gaussian per measurement.
    sigma = 1.
    use_as_noise_matrix = False # Is this the noise term? Solver needs to know.

    def _check_constraints(self):
        assert isinstance(self.sigma, float)
        assert isinstance(self.use_as_noise_matrix, bool)
        assert self.sigma>0
        super()._check_constraints()

    def get_number_of_parameters_from_mat_internal(self, mat):
        return mat.shape[0]

    def get_observable_relationship_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        prior_matrices.number_of_observations = mat.shape[0]      
        prior_matrices.observable_offset = np.zeros((mat.shape[0]))
        prior_matrices.design_matrix = sparse_matrix (sp.sparse.eye(mat.shape[0]))
        return prior_matrices

    def get_prior_distribution_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()        
        prior_matrices.number_of_parameters = mat.shape[0]
        prior_matrices.prior_mean = np.zeros((mat.shape[0]))        
        prior_matrices.prior_precision_matrix = sparse_matrix ((sp.sparse.eye(mat.shape[0]))/self.sigma**2)        
        if self.use_as_noise_matrix:
            prior_matrices.noise_parameter_indices = np.arange(mat.shape[0])
            assert not self.check_for_uniqueness # otherwise deal with reordering noise parameters
        return prior_matrices

    def get_prediction_from_mat_internal(self, mat):
        return self.parameters

    def update_hyperparameters_from_mat_internal(self,mat):
        return False

class UncorrelatedVaryingSigma(Uncorrelated):
    # sigma is now an array, each value of sigma applies to all observations for which features_hyper take a certain value
    # features_hyper must be specified as indices into features
    # used to handle different noise per wavelength
    sigma_varying = []
    features_hyper = []
    initialized_inds = []
    use_as_noise_matrix = False

    def initialize_from_mat_internal(self,mat):
        super().initialize_from_mat_internal(mat)
        _,self.initialized_inds = np.unique(mat[:,self.features_hyper],axis=0,return_inverse=True)       
        
        
    def get_prior_distribution_from_mat_internal(self,mat):
        prior_matrices = super().get_prior_distribution_from_mat_internal(mat)
        assert(self.initialized_inds.shape[0] == mat.shape[0])
        assert(np.max(self.initialized_inds) == self.sigma_varying.shape[0]-1)
        prior_matrices.prior_precision_matrix = sparse_matrix ( sp.sparse.diags( 1/(self.sigma_varying**2)[self.initialized_inds]))
        return prior_matrices

    def update_hyperparameters_from_mat_internal(self,mat):
        return False


class SquaredExponentialKernelMulti(FeatureSelector):
    # The Gaussian Process model. Combined multiple kernels with different length scale, each a squared-exponential kernel.
    sigmas = [] # np.array, one each for one length scale, and one additional one for noise
    lengths = [] # list of np.array, each specifying one length scale. Each one is an n-element array, where n is the number of dimensions (determined by the length of the features property)
    update_sigma = False # expose sigma values as hyperparameters?
    max_log_sigma_step = 25 # maximum update step for hyperparameter optimization
    maxiter = 10 # for sigma updates
    require_mean_of_non_noise_zero = False # doesn't do what you might expect for more than one feature
    

    def _check_constraints(self):
        assert isinstance(self.sigmas, np.ndarray)
        assert isinstance(self.lengths, list)
        assert self.sigmas.shape == (len(self.lengths)+1,)
        super()._check_constraints()

    def get_number_of_parameters_from_mat_internal(self, mat):
        return mat.shape[0]

    def get_observable_relationship_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        prior_matrices.number_of_observations = mat.shape[0]      
        prior_matrices.observable_offset = np.zeros((mat.shape[0]))
        prior_matrices.design_matrix = sparse_matrix (sp.sparse.eye(mat.shape[0]))
        return prior_matrices

    def get_K(self,sigmas,mat):
        # Helper function to get covariance matrix
        K = np.zeros((mat.shape[0], mat.shape[0]))
        for i in range(len(self.lengths)):
            # Add squared exponential for each length scale
            mat_normalized = mat/self.lengths[i]
            if mat_normalized.shape[0] == 0:
                distances = np.zeros((0,0))
            else:
                distances = sp.spatial.distance_matrix(mat_normalized, mat_normalized)
            K = K+sigmas[i]**2 * np.exp(-distances**2)
        if self.require_mean_of_non_noise_zero:
            # Make sure mean is zero
            if len(self.features)==1:
                J = np.ones((1,K.shape[0]))
                J[0] = 0.5
                J[-1] = 0.5
                K = K - K@(J.T@(sp.linalg.solve(J@K@J.T, (J@K))))
            else:
                assert len(self.features)==2
                base_mat = mat[:,1]
                _,inds = np.unique(base_mat,axis=0,return_inverse=True)
                n_obs = mat.shape[0]
                vals = np.ones(n_obs)
                vals[np.logical_or(base_mat==np.min(base_mat), base_mat==np.max(base_mat))] = 0.5
                J = sparse_matrix( (vals, (np.arange(n_obs), inds)) ).toarray().T
                Q = J@K@J.T
                K = K - K@(J.T@(sp.linalg.solve(Q+1e-3*Q[1,1]*np.eye(J.shape[0]), (J@K))))
        # Add noise (there must always be a small noise for numerical reasons)
        K = K+sigmas[-1]**2 * np.eye(mat.shape[0])
        return K
                          
         
    def get_prior_distribution_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        prior_matrices.number_of_parameters = mat.shape[0]
        prior_matrices.prior_mean = np.zeros((mat.shape[0]))

        K = self.get_K(self.sigmas, mat)

        prior_matrices.prior_precision_matrix =  sparse_matrix(sp.linalg.inv(K))
        return prior_matrices

    def get_prediction_from_mat_internal(self, mat):
        return self.parameters

    def update_hyperparameters_from_mat_internal(self,mat):
        if self.update_sigma:
            # Optimize log likelihood for the given parameters
            X = self.parameters
            def minus_log_likelihood(log_sigmas):
                K = self.get_K(np.exp(log_sigmas), mat)
                det_term = -0.5*np.linalg.slogdet(K).logabsdet
                exp_term = -0.5*X*(np.linalg.solve(K,X))
                return -(X.shape[1]*det_term + np.sum(exp_term))
            self.sigmas = np.exp(sp.optimize.fmin_l_bfgs_b(minus_log_likelihood, np.log(self.sigmas), approx_grad=True, \
                                                      epsilon=1e-3, disp=1, maxiter=self.maxiter, \
                                                      bounds = [(np.log(item)-self.max_log_sigma_step,np.log(item)+self.max_log_sigma_step) for item in self.sigmas])[0])
            fudge = 1e-3
            if self.sigmas[-1] < fudge*np.max(self.sigmas):
                self.sigmas[-1] = fudge*np.max(self.sigmas) # keep positive definite
            return True
        else:
            return False

            


def my_cholesky(P):
    # Finds (p,L,d) such that: P = d^-1*p'*L*L'*p*d^-1
    # p is a permutation matrix
    # L is lower triangular
    # d is diagonal
    factor = robust_cholesky(P)
    p = factor[0].P()
    p = sparse_matrix((np.ones(len(p), dtype=int), (np.arange(len(p)), p)), shape=(len(p),len(p)))
    L = factor[0].L()
    return p,L,factor[1]

def robust_cholesky(Q, start_fallback=0):
    # Returns two outputs:
    # 1: Choleskey factor object from sksparse.cholmod.cholesky
    # 2: A diagonal scaling matrix
    # See my_cholesky below for definitions
    try:
        assert start_fallback == 0
        diag_scaler = sparse_matrix(sp.sparse.eye(Q.shape[0]))
        res = cholesky(Q, mode="supernodal", ordering_method = "metis")
        return res, diag_scaler
    except:
        # Add increasingly large diagonals until Cholesky decomposition succeeds
        if start_fallback==0:
            scale=1e-10
        else:
            scale = start_fallback
        diag_scaler = sparse_matrix(sp.sparse.diags(0+1/np.sqrt(Q.diagonal())))
        Q = sparse_matrix(diag_scaler @ (Q @ diag_scaler))
        print('Cholesky fallback')
        max_eig = sp.sparse.linalg.eigs(Q,1)[0][0].real
        print('max_eig: ' + str(max_eig))
        while True:
            print('trying scale: '+ str(scale))
            try:
                res = cholesky(Q + scale * max_eig * sparse_matrix(sp.sparse.eye(Q.shape[0])), mode="supernodal", ordering_method = "metis")
                print('successful')
                return res, diag_scaler
            except:
                scale = scale*10
                if scale>0.01:
                    raise


def concatenate_sparse_csc_matrices(list_of_matrices):
    # Source: https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices/33259578#33259578
    num_rows = list_of_matrices[0].shape[0]
    num_cols = sum(m.shape[1] for m in list_of_matrices)

    data = []
    indices = []
    indptr_diff = [[0]]

    for m in list_of_matrices:
        if m.nnz > 0:
            data.append(m.data)
            indices.append(m.indices)
            indptr_diff.append(np.diff(m.indptr))
        else:
            indptr_diff.append([0] * m.shape[1])

    if data:
        data = np.concatenate(data)

    if indices:
        indices = np.concatenate(indices)

    indptr = np.cumsum(np.concatenate(indptr_diff))
    concatenated_matrix = sparse_matrix((data, indices, indptr), shape=(num_rows, num_cols))

    return concatenated_matrix

def find_parameters_to_make_noise_consistent(model, obs, prior_matrices):
    # Makes sure the parameters in model lead to predicitions consistent with the labels in obs, by changing the noise
    noise_parameter_indices = prior_matrices.noise_parameter_indices

    pred = model.get_prediction(obs)
    offset = obs.labels - pred
    params = model.get_parameters()
    params[noise_parameter_indices,:] = params[noise_parameter_indices,:] + offset    
    if kgs.debugging_mode >= 2:
        model2 = copy.deepcopy(model)
        model2.set_parameters(params)
        pred2 = model2.get_prediction(obs)
        assert np.all(np.abs(pred2-obs.labels)<=1e-10*np.abs(obs.labels))   
    return params

def log_likelihood(model, obs, cholQ=None, get_value=True, get_gradients=False):
    # Compute p(obs.labels|model)
    # Inputs:
    # -model: see equation
    # -obs: see equation
    # -choLQ: precomputed Cholesky decomposition
    # -get_value: whether to return the value of the of the log likelihood
    # -get_gradients: whether to return the gradients of the log likelihood to the hyperparameters of the model
    # The inner workings of this function are not explained.
    assert model.number_of_instances == 1
    assert obs.number_of_instances == 1

    prior_matrices = model.get_prior_matrices(obs)

    Pfull = prior_matrices.prior_precision_matrix
    J = prior_matrices.design_matrix
    y = ((obs.labels-J@np.reshape(prior_matrices.prior_mean,(-1,1))).T - prior_matrices.observable_offset ).T
    noise_indices = prior_matrices.noise_parameter_indices
    all_indices = np.arange(prior_matrices.number_of_parameters)
    non_noise_indices = np.setdiff1d(all_indices, noise_indices, assume_unique = True)

    N = Pfull
    N = Pfull[noise_indices,:][:,noise_indices]
    P = Pfull
    P = P[non_noise_indices,:][:,non_noise_indices]
    J = J[:,non_noise_indices]
 
    if cholQ == None:
        Q = P+J.T@N@J;
        cholQ = robust_cholesky(Q);
    cholP = robust_cholesky(P);    

    rhs = J.T@(N@y)
    mu = cholQ[1]@(cholQ[0].solve_A(cholQ[1]@rhs))
    res = y - J@mu

    if get_value:
        cholN = robust_cholesky(N);
        value = 0
        value = value + 0.5*cholN[0].logdet() - np.sum(np.log(cholN[1].diagonal()))
        value = value - 0.5*(res.T@N@res)
        value = value + 0.5*cholP[0].logdet() - np.sum(np.log(cholP[1].diagonal()))
        value = value - 0.5*(mu.T@P@mu)
        value = value - 0.5*cholQ[0].logdet() + np.sum(np.log(cholQ[1].diagonal()))
        value = value[0,0]
    else:
        value = None

    # Gradients
    if get_gradients:
        P_derivs = model.get_partial_prior_precision_matrices(obs)    
        gradients = np.zeros((model.number_of_hyperparameters))
        for i in range(model.number_of_hyperparameters):
            chi = np.random.default_rng(seed=0).integers(0,high=1,size=(P.shape[0], 100),endpoint=True)*2-1

            P_diff = P_derivs[i]
            P_diff = P_diff[non_noise_indices,:][:,non_noise_indices]       
            mu_diff = -cholQ[1]@(cholQ[0].solve_A(cholQ[1]@(P_diff @ mu)))
            res_diff = -J@mu_diff
    
            value_diff = 0
            value_diff = value_diff - (res.T@N@res_diff)
            value_diff = value_diff + 0.5* np.sum(chi*(cholP[1]@(cholP[0].solve_A(cholP[1]@(P_diff@chi)))))/chi.shape[1]         
            value_diff = value_diff - (mu.T@P@mu_diff) - 0.5*(mu.T@P_diff@mu)
            value_diff = value_diff - 0.5* np.sum(chi*(cholQ[1]@(cholQ[0].solve_A(cholQ[1]@P_diff@chi))))/chi.shape[1]
    
            gradients[i] = value_diff
    else:
        gradients = None
        

    return value, gradients


def sample_from_prior(model, obs, rng=None, n_samples=0):
    # Take samples from the prior defined by model, and put them in obs
    if rng is None:
        rng = np.random.default_rng(seed=0)

    model = copy.deepcopy(model)
    prior_matrices = model.get_prior_matrices(obs)
    Q = prior_matrices.prior_precision_matrix
    p,L,d = my_cholesky(Q)
    uncorr = rng.normal(size=(prior_matrices.number_of_parameters, n_samples))
    corr = d@(p.T@(sp.sparse.linalg.spsolve(L.T, uncorr)))
    corr = (corr.T + prior_matrices.prior_mean).T
    if n_samples==1:
        corr = np.reshape(corr, (-1,1))
    model.set_parameters(corr)

    return model

#@kgs.profile_each_line
def solve_gp(model, obs, rng=None, n_samples=0, fill_noise_parameters=True, clear_caches=False):   
    # Solves the GP assuming the model is linear.
    # Inputs:
    # -model: model to solve
    # -obs: observable defining the right-hand side
    # -rng: random number generator, used if samples are requested
    # -n_samples: number of samples to generate
    # -fill_noise_parameters: whether to make the noise parameters of the model consistent (takes extra time)
    # -clear_caches: whether to clear all prior matrix caches before outputting
    # Outputs:
    # -model: mean of the posterior
    # -model_samples: samples from the posterior (not in the output if n_samples=0)
    # -choLQ: Cholesky decmposition computed along the way, can be useful for log_likelihood()
    # The inner workings of this function are not explained.
    
    if rng is None:
        rng = np.random.default_rng(seed=0)
    model.check_constraints(debugging_mode_offset = 1)
    obs.check_constraints(debugging_mode_offset = 1)
    if n_samples>0:
        assert obs.number_of_instances==1
    else:
        assert obs.number_of_instances>0    
    prior_matrices = model.get_prior_matrices(obs)
    if clear_caches:
        model.clear_all_caches()
    model = copy.deepcopy(model)
    P = prior_matrices.prior_precision_matrix
    J = prior_matrices.design_matrix
    mu = np.reshape(prior_matrices.prior_mean, (-1,1))
    o = prior_matrices.observable_offset
    labels_with_offset = ((obs.labels-J@mu).T - o ).T

    if np.any(np.isnan(labels_with_offset)):
        raise kgs.ArielException(5, 'NaNs in GP solve labels')
    
    # Find noise
    noise_indices = prior_matrices.noise_parameter_indices
    N = P[noise_indices,:][:, noise_indices]
    assert N.count_nonzero() == prior_matrices.number_of_observations
    scaling_factor = np.sqrt(N.diagonal())
    all_indices = np.arange(prior_matrices.number_of_parameters)
    non_noise_indices = np.setdiff1d(all_indices, noise_indices, assume_unique = True)
    
    J = J[:,non_noise_indices]
    P = P[non_noise_indices,:][:, non_noise_indices]    
    Js = sp.sparse.diags(scaling_factor) @ J

    Q = (Js.T@Js+P) # not faster on GPU
    cholQ = robust_cholesky(Q)
    meanPart = cholQ[1]@cholQ[0].solve_A(cholQ[1]@( J.T@(N@labels_with_offset) ))
    if len(meanPart.shape)==1:
        meanPart = np.reshape(meanPart, (-1,1))

    mean = np.zeros((model.number_of_parameters, obs.number_of_instances))    
    mean[non_noise_indices,:] = meanPart
    mean = mean + mu
    if fill_noise_parameters:
        model.set_parameters(mean)
        mean = find_parameters_to_make_noise_consistent(model, obs, prior_matrices)

    if np.any(np.isnan(mean)):
        raise kgs.ArielException(5, 'NaNs in GP solve output')

    model.set_parameters(mean)
    
    if n_samples == 0:
        return model, cholQ
    else:
        p = cholQ[0].P()
        p = sparse_matrix((np.ones(len(p), dtype=int), (np.arange(len(p)), p)), shape=(len(p),len(p)))
        L = cholQ[0].L()  
        uncorr = rng.normal(size=(meanPart.shape[0], n_samples))
        corr = cholQ[1]@(p.T@sp.sparse.linalg.spsolve_triangular(L.T, uncorr, lower=False, overwrite_A=True, overwrite_b=True))
        corr = (corr.T + meanPart[:,0]).T
        samples = np.zeros((model.number_of_parameters, n_samples))
        samples[non_noise_indices,:] = corr
        samples = samples + mu
        model_samples = copy.deepcopy(model)
        if fill_noise_parameters:            
            model_samples.set_parameters(samples)
            samples = find_parameters_to_make_noise_consistent(model_samples, obs, prior_matrices)
        model_samples.set_parameters(samples)
        if kgs.debugging_mode >= 2 and model.is_linear():
            labels = model_samples.get_prediction(obs)
            assert np.all(np.abs(labels-obs.labels)<=1e-8*np.abs(labels))
        return model, model_samples, cholQ

#@kgs.profile_each_line
def solve_gp_nonlinear(model, obs, rng=None, n_samples=0, n_samples_mle=10, n_iter = 25, update_rate = 0.1, update_hyperparameters_from=10, hyperparameter_method = 'em', max_log_update_initial=1., fill_noise_parameters=True, adapt_func = lambda x:x):
    # Solves a GP model iteratively, each time linearizing around the mean of the posterior of the previous step. Also updates hyperparameters along the way.
    
    assert hyperparameter_method=='em' or hyperparameter_method=='gradient_descent'
    model = copy.deepcopy(model)    
    if rng is None:
        rng = np.random.default_rng(seed=0)
    state = copy.deepcopy(rng.bit_generator.state)
    for i in range(n_iter):
        #print(model.m['signal'].m['main'].m['transit'].get_parameters()[-model.m['signal'].m['main'].m['transit'].number_of_extra_parameters:].T)
        rng.bit_generator.state = copy.deepcopy(state)   
        if i<update_hyperparameters_from:
            # Iteration with just parameter updates
            model_new = solve_gp(model, obs, fill_noise_parameters=False)[0]
        elif hyperparameter_method == 'em':
            # Iteration with expectation maximization based on posterior samples
            model_new, model_samples, _ = solve_gp(model, obs, rng=rng, n_samples=n_samples_mle, fill_noise_parameters=False)
            model_samples.update_hyperparameters()

            hparam_old = model.get_hyperparameters()
            hparam_new = model_samples.get_hyperparameters()
            model.set_hyperparameters(hparam_new)
            print('Hyperparameters from ', hparam_old, ' to ',hparam_new)
        else:
            # Iteration with gradient descent on minus log likelihood
            # Fit
            model_new, cholQ = solve_gp(model, obs, fill_noise_parameters=False)
            
            # Hyperparameter update
            hparam_old = model.get_hyperparameters()
            if i==update_hyperparameters_from:
                 max_log_update = max_log_update_initial * np.ones(hparam_old.shape)
            _,gradient_nonlog = log_likelihood(model,obs, get_gradients=True, get_value=False, cholQ=cholQ)
            gradient = hparam_old * gradient_nonlog
            
            hparam_step = 4*gradient            
            if i>update_hyperparameters_from:
                for jj in range(len(hparam_old)):
                    if not np.sign(hparam_step[jj]) == np.sign(prev_step[jj]):
                        max_log_update[jj] = max_log_update[jj]/2
            hparam_step[hparam_step>max_log_update] = max_log_update[hparam_step>max_log_update]
            hparam_step[hparam_step<-max_log_update] = -max_log_update[hparam_step<-max_log_update]            
            log_hparam_new = np.log(hparam_old) + hparam_step            
            prev_step = hparam_step
            hparam_new = np.exp(log_hparam_new) 
            model.set_hyperparameters(hparam_new)          
                
        # Parameter update
        #print('before', log_likelihood_given_parameters(model)
        param_old = model.get_parameters()
        param_new = param_old + update_rate*(model_new.get_parameters()-param_old)
        #print('after', log_likelihood_given_parameters(model)
        
        model.set_parameters(param_new)
        
 

    rng.bit_generator.state = copy.deepcopy(state)
    # Adapt the final model as specified by the caller, typically modifying hyperparameters
    model = adapt_func(model)
    # Final fit, generating the requested number of samples
    a,b,_ = solve_gp(model, obs, rng=rng, n_samples=n_samples, fill_noise_parameters=fill_noise_parameters, clear_caches = True)
    return a,b


class CustomKernel(FeatureSelector):
    kernel_types = None
    x_forms = None
    hyperparameters = None

    def _check_constraints(self):
        # todo
        super()._check_constraints()

    def get_number_of_parameters_from_mat_internal(self, mat):
        return mat.shape[0]

    def get_observable_relationship_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        prior_matrices.number_of_observations = mat.shape[0]      
        prior_matrices.observable_offset = np.zeros((mat.shape[0]))
        prior_matrices.design_matrix = sparse_matrix (sp.sparse.eye(mat.shape[0]))
        return prior_matrices
    
    def K_matrix(self, mat):
        hyperparameters = self.hyperparameters
        kernel_types = self.kernel_types
        x_forms = self.x_forms
        xx = mat
        #print(x.shape)
        try: hyperparameters = hyperparameters.get()
        except: pass
        cur_ind = 0

        # noise
        K = hyperparameters[cur_ind]**2*cp.eye(xx.shape[0])
        #K = 1e-6**2*cp.eye(x.shape[0])
        cur_ind+=1

        for i_kernel in range(len(kernel_types)):        
            #xx = x[:,None]
            if x_forms[i_kernel] == 'id':
                this_x = xx
            elif x_forms[i_kernel] == 'gamma':
               # print('gamma', hyperparameters[cur_ind], cur_ind)
                this_x = xx**hyperparameters[cur_ind]
                cur_ind+=1
            elif x_forms[i_kernel] == 'log':
                this_x = cp.log(xx)
            else:
                raise 'stop'

            sigma = hyperparameters[cur_ind]
            cur_ind+=1

            sigma_mat = sigma**2
            
            if kernel_types[i_kernel] == 'SE':     
                ell = hyperparameters[cur_ind]
                cur_ind+=1
                K += sigma_mat * cp.exp(-(this_x - this_x.T)**2 / (2 * ell**2))
            elif kernel_types[i_kernel] == 'SE_cosine':     
                ell = hyperparameters[cur_ind]
                p = hyperparameters[cur_ind+1]
                cur_ind+=2
                distances = (this_x - this_x.T)
                K += sigma_mat * cp.exp(-distances**2 / (2 * ell**2)) * cp.cos(2*cp.pi*distances/ell*p)
            elif kernel_types[i_kernel] == 'matern12':
                ell = hyperparameters[cur_ind]
                cur_ind+=1
                K += sigma_mat * cp.exp(-cp.abs(this_x - this_x.T) / ell)
            elif kernel_types[i_kernel] == 'matern32':
                ell = hyperparameters[cur_ind]
                cur_ind+=1
                r = cp.abs(this_x - this_x.T) / ell
                a = cp.sqrt(3.0) * r
                K += sigma_mat * (1.0 + a) * cp.exp(-a)
            elif kernel_types[i_kernel] == 'matern52':
                ell = hyperparameters[cur_ind]
                cur_ind+=1
                r = cp.abs(this_x - this_x.T) / ell
                a = cp.sqrt(5.0) * r
                K += sigma_mat * (1.0 + a + (5.0 / 3.0) * r**2) * cp.exp(-a)
            elif kernel_types[i_kernel] == 'old_fixed':
                K = sigma_mat*cp.array(get_K_mm(mm, mm.sigmas, this_x.get()))
            elif kernel_types[i_kernel] == 'old_dynamic':
                mmm = copy.deepcopy(mm)
                mmm.sigmas[:-1] = hyperparameters[cur_ind:cur_ind+9]
                cur_ind+=9
                K = cp.array(get_K_mm(mmm, mmm.sigmas, this_x.get()))
            else:
                raise['stop']

        assert cur_ind == len(hyperparameters)
        return K

         
    def get_prior_distribution_from_mat_internal(self,mat):
        prior_matrices = PriorMatrices()
        prior_matrices.number_of_parameters = mat.shape[0]
        prior_matrices.prior_mean = np.zeros((mat.shape[0]))

        K = self.K_matrix(cp.array(mat)).get()

        prior_matrices.prior_precision_matrix =  sparse_matrix(sp.linalg.inv(K))
        return prior_matrices

    def get_prediction_from_mat_internal(self, mat):
        return self.parameters
