'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module implements some helpers for modeling, supporting the main modeling flow in ariel_gp.py
'''

import kaggle_support as kgs
import ariel_gp
import ariel_pca
from dataclasses import dataclass, field
import scipy
import numpy as np
import cupy as cp
import copy
import time

def baseline_model():
    # Defines the default model
    
    # 3 layers of helpers:
    # - MultiTransit averages over multiple transits
    # - Fudget applies several tweaks to the predictions, chosen to optimize the score on the training set
    # - PCA handles PCA decomposition of the training lables    
    model = MultiTransit(model=Fudger(model=ariel_pca.PCA(model=ariel_gp.PredictionModel())))
    model.model.model.model_options_link = model.model.model.model.model_options # make sure PCA knows where to put its outputs
    model.model.model.model.run_in_parallel = True # run the model in parallel to speed up
    return model

@dataclass
class Fudger(kgs.Model):
    # Applies several tweaks to the predictions of the underlying model
    
    model: kgs.Model = field(init=True, default=None) # The underlying model
    
    # Adapt mean of prediction for FGS and AIRS separately:
    # output = bias_a + bias_b * input + adjust_based_on_u*u[0], where u[0] is the first limb darkening parameter
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])    
    adjust_based_on_u: list = field(init=True, default_factory=lambda:[0.,0.])
    do_adjust_based_on_u = True
    
    # Adjust sigma prediction; see code for exact equations
    sigma_offset: list = field(init=True, default_factory=lambda:[0.,0.]) # Apply an offset to FGS and AIRS
    sigma_fudge_FGS = 1. # Apply a scaling to FGS 
    sigma_fudge_AIRS_mean = 1. # Apply a scaling to the mean of AIRS
    sigma_fudge_AIRS_var = 1. # Apply a scaling to the variation of AIRS    
    sigma_fudge_based_on_AIRS_var: list = field(init=True, default_factory=lambda:[0.,0.]) # Apply additional sigma if AIRS variation is high
    fudge_based_on_AIRS_var = True
    
    # Internal helpers
    _cached_planet_id = None
    _cached_result = None
    _disable_transforms = False
    _AIRS_var_values = None
    _FGS_u = None
    _AIRS_u = None
    
    def _to_x(self):
        # Convert parameters to vector, for use in scipy.optimize
        the_list = self.bias_a + self.bias_b + self.sigma_offset + [self.sigma_fudge_FGS, self.sigma_fudge_AIRS_mean, self.sigma_fudge_AIRS_var] + self.sigma_fudge_based_on_AIRS_var + self.adjust_based_on_u
        return np.reshape(the_list, (-1,))
    def _from_x(self,x):
        # Inverse of above
        self.bias_a[0] = x[0]
        self.bias_a[1] = x[1]
        self.bias_b[0] = x[2]
        self.bias_b[1] = x[3]
        self.sigma_offset[0] = x[4]
        self.sigma_offset[1] = x[5]
        self.sigma_fudge_FGS = x[6]
        self.sigma_fudge_AIRS_mean = x[7]
        self.sigma_fudge_AIRS_var = x[8]
        self.sigma_fudge_based_on_AIRS_var[0] = x[9]
        self.sigma_fudge_based_on_AIRS_var[1] = x[10]
        self.adjust_based_on_u[0] = x[11]
        self.adjust_based_on_u[1] = x[12]
        assert np.all(self._to_x() == x)
        
    def alter_mats(self, mats):
        # Apply all the fudges. This is done on matrix versions of the predictions for speed.
        
        # Adapt mean of prediction
        y_true,y_pred,cov_pred = mats
        y_pred[:,0] *= self.bias_a[0]
        y_pred[:,0] += self.bias_b[0]
        y_pred[:,1:] *= self.bias_a[1]
        y_pred[:,1:] += self.bias_b[1]        
        if self.do_adjust_based_on_u:
            y_pred[:,0] += self.adjust_based_on_u[0] * self._FGS_u
            y_pred[:,1:] += self.adjust_based_on_u[1] * self._AIRS_u[:,None]
        
            
        # Adapt sigma
        cov_pred[:,0,0] += self.sigma_offset[0]**2
        cov_pred[:,1:,1:] += self.sigma_offset[1]**2
        cov_pred[:,0,:] *= self.sigma_fudge_FGS
        cov_pred[:,:,0] *= self.sigma_fudge_FGS
        AIRS_mean = np.mean(cov_pred[:,1:,1:], axis=(1,2))
        AIRS_var = cov_pred[:,1:,1:]-AIRS_mean[:,None,None]
        AIRS_mean *= self.sigma_fudge_AIRS_mean**2
        AIRS_var *= self.sigma_fudge_AIRS_var**2
        cov_pred[:,1:,1:] = AIRS_var+AIRS_mean[:,None,None]        
        if self.fudge_based_on_AIRS_var:
            cov_pred[:,0,0] += self.sigma_fudge_based_on_AIRS_var[0]**2*self._AIRS_var_values**2
            idx = np.arange(1, 283)
            cov_pred[:,idx,idx] += (self.sigma_fudge_based_on_AIRS_var[1]**2*self._AIRS_var_values**2)[:,None]
    
    def _train(self,train_data):
        
        # Train underlying model
        self.model.train(train_data)
          
        # Infer data with underlying model, and convert to matrices for faster handling
        self.state=1    
        self._disable_transforms = True # don't apply fudge just yet
        inferred_data = self.infer(train_data)
        self._disable_transforms = False
        mats = kgs.data_to_mats(inferred_data,train_data)
        
        # Optimize fudge parameters
        def cost(x):            
            self._from_x(x)
            mats_here = copy.deepcopy(mats)
            self.alter_mats(mats_here)
            res =-kgs.score_metric_fast(*mats_here)
            return res
        x0 = self._to_x()
        res = scipy.optimize.minimize(cost,x0,tol=1e-2)
        self._from_x(res.x)
        
    
    def _infer(self,data):
        
        # Infer underlying model, using cache if possible
        if self._cached_planet_id is not None and [d.planet_id for d in data]==self._cached_planet_id:
            data = copy.deepcopy(self._cached_result)
        else:
            data = self.model.infer(data)
            if self._cached_planet_id is None:
                # Cache results in case we are training and inferring on the same data
                self._cached_result = copy.deepcopy(data)
                self._cached_planet_id = [d.planet_id for d in data]
                
        # Apply fudges
        mats = kgs.data_to_mats(data,data)
        self._AIRS_var_values = cp.array([np.std(d.spectrum) for d in data])
        self._FGS_u = cp.array([d.diagnostics['transit_params_gp'][0].u[0] for d in data])
        self._AIRS_u = cp.array([d.diagnostics['transit_params_gp'][1].u[0] for d in data])        
        if not self._disable_transforms:
            self.alter_mats(mats)
        kgs.mats_to_data(data,copy.deepcopy(data),mats)
            
        return data

@dataclass
class MultiTransit(kgs.Model):
    # Models multiple transits and averages them
    
    model: kgs.Model = field(init=True, default=None) # underlying model
    variance_fudge = 1.4 # Fudge to apply to variance (error often correlates between transits)
    
    def _convert_data(self, data):
        # Make one list of all the transits to pass to the underlying model
        data_run = []
        for d in data:
            if len(d.transits)==1:
                data_run.append(d)
            else:
                dd = copy.deepcopy(d)
                dd.transits = dd.transits[:1]
                data_run.append(dd)
                dd = copy.deepcopy(d)
                dd.transits = dd.transits[1:]
                data_run.append(dd)
        return data_run
        
    def _train(self,train_data):
        # Train underlying model
        self.model.train(train_data)
    
    def _infer(self,data):        
        
        # Run underlying model on each transit
        data_run = self._convert_data(data) 
        data_run = self.model.infer(data_run)
        
        # Combine data
        planet_ids_orig = np.array([d.planet_id for d in data])
        planet_ids_run = np.array([d.planet_id for d in data_run])
        data_output = []
        for d in data:
            if len(d.transits)==1:
                # 1 transit planet: just use output of underlying model
                ind = np.argwhere(d.planet_id == planet_ids_run)
                assert(len(ind)==1)
                data_output.append(data_run[ind[0][0]])
            else:
                # 2 transit planet: average transits. Note that the equations here are only really correct if the uncertainty prediction is the same for both transits.
                ind = np.argwhere(d.planet_id == planet_ids_run)    
                assert(len(ind)==2)
                this_data1 = data_run[ind[0][0]]
                this_data2 = data_run[ind[1][0]]           
                this_data1.spectrum = this_data1.spectrum/2 + this_data2.spectrum/2
                this_data1.spectrum_cov = self.variance_fudge * (this_data1.spectrum_cov/4 + this_data2.spectrum_cov/4)
                data_output.append(this_data1)
                
        return data_output