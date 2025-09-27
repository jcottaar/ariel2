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
    
    # Adapt mean of prediction for FGS and airs separately:
    # output = bias_a + bias_b * input + adjust_based_on_u*u[0], where u[0] is the first limb darkening parameter
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])    
    adjust_based_on_u: list = field(init=True, default_factory=lambda:[0.,0.])
    do_adjust_based_on_u = True
    
    sigma_offset: list = field(init=True, default_factory=lambda:[0.,0.])
    sigma_fudge_FGS = 1.
    sigma_fudge_AIRS_mean = 1.
    sigma_fudge_AIRS_var = 1.
    
    sigma_fudge_based_on_AIRS_var: list = field(init=True, default_factory=lambda:[0.,0.])
    fudge_based_on_AIRS_var = True
    
    
    
    sigma_fudge_multi = 1.
    do_fudge_multi = False

    
    
    
    _cached_planet_id = None
    _cached_result = None
    _disable_transforms = False
    _AIRS_var_values = None
    _FGS_u = None
    _AIRS_u = None
    _is_multi_transit = None
    
    def _to_x(self):
        the_list = self.bias_a + self.bias_b + self.sigma_offset + [self.sigma_fudge_FGS, self.sigma_fudge_AIRS_mean, self.sigma_fudge_AIRS_var] + self.sigma_fudge_based_on_AIRS_var + self.adjust_based_on_u + [self.sigma_fudge_multi]
        return np.reshape(the_list, (-1,))
    def _from_x(self,x):
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
        self.sigma_fudge_multi = x[13]
        assert np.all(self._to_x() == x)
        
    def alter_mats(self, mats):
        y_true,y_pred,cov_pred = mats
        y_pred[:,0] *= self.bias_a[0]
        y_pred[:,0] += self.bias_b[0]
        y_pred[:,1:] *= self.bias_a[1]
        y_pred[:,1:] += self.bias_b[1]
        
        if self.do_adjust_based_on_u:
            y_pred[:,0] += self.adjust_based_on_u[0] * self._FGS_u
            y_pred[:,1:] += self.adjust_based_on_u[1] * self._AIRS_u[:,None]
        
            
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
            
        if self.do_fudge_multi:
            cov_pred[self._is_multi_transit,:,:] *= self.sigma_fudge_multi
        #cov_pred[:,1:,1:] *= self.sigma_fudge_AIRS_mean**2
        #cov_pred[:,:,1:] *= self.sigma_fudge_AIRS_mean
    
    def _train(self,train_data):
        self.model.train(train_data)
        self.state=1
        ii=0
        t=time.time()
        
        self._disable_transforms = True
        inferred_data = self.infer(train_data)
        self._disable_transforms = False
        #print(kgs.score_metric(inferred_data,train_data))
        mats = kgs.data_to_mats(inferred_data,train_data)
        
        #@kgs.profile_each_line
        def cost(x):
            
            self._from_x(x)
            #inferred_data = self.infer(train_data)
            mats_here = copy.deepcopy(mats)
            self.alter_mats(mats_here)
            nonlocal  ii
            ii+=1
            #print(ii)
            res =-kgs.score_metric_fast(*mats_here)
            #print(res)
            return res
        x0 = self._to_x()
        res = scipy.optimize.minimize(cost,x0,tol=1e-2)
        self._from_x(res.x)
        print('Opt time', time.time()-t)
        
    
    def _infer(self,data):
        self._is_multi_transit = cp.array([len(d.transits)>1 for d in data])
        if self._cached_planet_id is not None and [d.planet_id for d in data]==self._cached_planet_id:
            data = copy.deepcopy(self._cached_result)
        else:
            data = self.model.infer(data)
            if self._cached_planet_id is None:
                self._cached_result = copy.deepcopy(data)
                self._cached_planet_id = [d.planet_id for d in data]
                
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
    
    model: kgs.Model = field(init=True, default=None)
    variance_fudge = 1.4
    
    def _convert_data(self, data):
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
        self.model.train(train_data)
        self.state=1
        
    
    def _infer(self,data):        
        
        data_run = self._convert_data(data)
        
        data_run = self.model.infer(data_run)
        
        planet_ids_orig = np.array([d.planet_id for d in data])
        planet_ids_run = np.array([d.planet_id for d in data_run])
        data_output = []
        for d in data:
            if len(d.transits)==1:
                ind = np.argwhere(d.planet_id == planet_ids_run)
                assert(len(ind)==1)
                data_output.append(data_run[ind[0][0]])
            else:
                ind = np.argwhere(d.planet_id == planet_ids_run)    
                assert(len(ind)==2)
                this_data1 = data_run[ind[0][0]]
                this_data2 = data_run[ind[1][0]]
                #this_data1.spectrum, this_data1.spectrum_cov = combine_measurements(this_data1.spectrum, this_data1.spectrum_cov, this_data2.spectrum, this_data2.spectrum_cov)             
                this_data1.spectrum = this_data1.spectrum/2 + this_data2.spectrum/2
                this_data1.spectrum_cov = self.variance_fudge * (this_data1.spectrum_cov/4 + this_data2.spectrum_cov/4)
                data_output.append(this_data1)
                
        return data_output

@dataclass
class SanityWrapper(kgs.Model):
    
    model: kgs.Model = field(init=True, default=None)
    action = 'do_nothing'
    max_errors = 1
    
    logged_errors: list = field(init=True, default_factory=list)
        
    def _train(self,train_data):
        self.model.train(train_data)
        self.state=1
        
    def infer(self,test_data):
        if self.action == 'explode':
            print('EXPLODING!!!!!!')
        inferred_data = super().infer(test_data)
        for d in inferred_data:
            if 'logged_error' in d.diagnostics.keys():
                self.logged_errors.append(d.diagnostics['logged_error'])
                
        if len(self.logged_errors)>self.max_errors:
            raise kgs.ArielException(len(self.logged_errors)/2, 'Too many diagnostics failures')
                
        all_spectra = np.array([d.spectrum for d in inferred_data])
        spectrum_mean = np.nanmean(all_spectra,0)
        spectrum_std = np.nanstd(all_spectra,0)
        print(spectrum_mean.shape, spectrum_std.shape)
        for d in inferred_data:
            if np.isnan(d.spectrum[0]):
                assert self.action == 'replace_by_mean'
                d.spectrum = spectrum_mean
                d.spectrum_cov = np.diag(spectrum_std**2)
        return inferred_data
        
    
    def _infer_single(self,data):        
        
        try:
            kgs.sanity_checks_active = True
            kgs.sanity_checks_without_errors = False
            return self.model.infer([data])[0]
        except kgs.ArielException as err:
            data.diagnostics['logged_error'] = err
            match self.action:
                case 'do_nothing':
                    kgs.sanity_checks_without_errors = True
                    return self.model.infer([data])[0]
                case 'explode':
                    raise 'BLOCKED'
                    data.spectrum = 1e8*np.ones(283)
                    data.spectrum_cov = 1e8*np.eye(283)
                    return data
                case 'replace_by_mean':
                    data.spectrum = np.nan*np.ones(283)
                    data.spectrum_cov = np.nan*np.eye(283)
                    return data
                case 'raise':
                    raise err
                case _:
                    raise Exception('Bad action')
    


