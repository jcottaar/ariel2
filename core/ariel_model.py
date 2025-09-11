import kaggle_support as kgs
import ariel_simple
import ariel_gp
from dataclasses import dataclass, field, fields
import scipy
import numpy as np
import cupy as cp
import copy
import time

@dataclass
class Fudger(kgs.Model):
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])
    sigma_fudge : list = field(init=True, default_factory=lambda:[1.,1.])
    sigma_offset: list = field(init=True, default_factory=lambda:[0.,0.])
    
    include_sigma_offset: bool = field(init=True, default=False)
    
    model: kgs.Model = field(init=True, default=None)
    
    _cached_planet_id = None
    _cached_result = None
    
    def _to_x(self):
        the_list = self.bias_a + self.bias_b + self.sigma_fudge
        if self.include_sigma_offset:
            the_list += self.sigma_offset
        return np.reshape(the_list, (-1,))
    def _from_x(self,x):
        self.bias_a[0] = x[0]
        self.bias_a[1] = x[1]
        self.bias_b[0] = x[2]
        self.bias_b[1] = x[3]
        self.sigma_fudge[0] = x[4]
        self.sigma_fudge[1] = x[5]
        if self.include_sigma_offset:
            self.sigma_offset[0] = x[6]
            self.sigma_offset[1] = x[7]
        assert np.all(self._to_x() == x)
        
    def alter_mats(self, mats):
        y_true,y_pred,cov_pred = mats
        y_pred[:,0] *= self.bias_a[0]
        y_pred[:,0] += self.bias_b[0]
        y_pred[:,1:] *= self.bias_a[1]
        y_pred[:,1:] += self.bias_b[1]
        cov_pred[:,0,0] += self.sigma_offset[0]**2
        cov_pred[:,1:,1:] += self.sigma_offset[1]**2
        cov_pred[:,0,:] *= self.sigma_fudge[0]
        cov_pred[:,:,0] *= self.sigma_fudge[0]
        cov_pred[:,1:,:] *= self.sigma_fudge[1]
        cov_pred[:,:,1:] *= self.sigma_fudge[1]
    
    def _train(self,train_data):
        self.model.train(train_data)
        self.state=1
        ii=0
        self.infer(train_data) # dummy
        t=time.time()
        
        inferred_data = self.infer(train_data)
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
            return -kgs.score_metric_fast(*mats_here)
        x0 = self._to_x()
        res = scipy.optimize.minimize(cost,x0,tol=1e-2)
        self._from_x(res.x)
        print('Opt time', time.time()-t)
        
    
    def _infer(self,data):
        if not self._cached_planet_id is None and [d.planet_id for d in data]==self._cached_planet_id:
            data = copy.deepcopy(self._cached_result)
        else:
            data = self.model.infer(data)
            if self._cached_planet_id is None:
                self._cached_result = copy.deepcopy(data)
                self._cached_planet_id = [d.planet_id for d in data]
                
        mats = kgs.data_to_mats(data,data)
        self.alter_mats(mats)
        kgs.mats_to_data(data,copy.deepcopy(data),mats)
            
        # for d in data:
        #     d.spectrum[0] *= self.bias_a[0]
        #     d.spectrum[0] += self.bias_b[0]
        #     d.spectrum[1:] *= self.bias_a[1]
        #     d.spectrum[1:] += self.bias_b[1]
        #     d.spectrum_cov[0,:] *= self.sigma_fudge[0]
        #     d.spectrum_cov[:,0] *= self.sigma_fudge[0]
        #     d.spectrum_cov[1:,:] *= self.sigma_fudge[1]
        #     d.spectrum_cov[:,1:] *= self.sigma_fudge[1]
            
        return data
    
    
@dataclass
class Fudger2(kgs.Model):
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])    
    sigma_offset: list = field(init=True, default_factory=lambda:[0.,0.])
    sigma_fudge_FGS = 1.
    sigma_fudge_AIRS_mean = 1.
    sigma_fudge_AIRS_var = 1.

    
    model: kgs.Model = field(init=True, default=None)
    
    _cached_planet_id = None
    _cached_result = None
    _disable_transforms = False
    
    def _to_x(self):
        the_list = self.bias_a + self.bias_b + self.sigma_offset + [self.sigma_fudge_FGS, self.sigma_fudge_AIRS_mean, self.sigma_fudge_AIRS_var]        
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
        assert np.all(self._to_x() == x)
        
    def alter_mats(self, mats):
        y_true,y_pred,cov_pred = mats
        y_pred[:,0] *= self.bias_a[0]
        y_pred[:,0] += self.bias_b[0]
        y_pred[:,1:] *= self.bias_a[1]
        y_pred[:,1:] += self.bias_b[1]
        cov_pred[:,0,0] += self.sigma_offset[0]**2
        cov_pred[:,1:,1:] += self.sigma_offset[1]**2
        cov_pred[:,0,:] *= self.sigma_fudge_FGS
        cov_pred[:,:,0] *= self.sigma_fudge_FGS
        AIRS_mean = np.mean(cov_pred[:,1:,1:], axis=(1,2))
        AIRS_var = cov_pred[:,1:,1:]-AIRS_mean[:,None,None]
        AIRS_mean *= self.sigma_fudge_AIRS_mean**2
        AIRS_var *= self.sigma_fudge_AIRS_var**2
        cov_pred[:,1:,1:] = AIRS_var+AIRS_mean[:,None,None]
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
        if not self._cached_planet_id is None and [d.planet_id for d in data]==self._cached_planet_id:
            data = copy.deepcopy(self._cached_result)
        else:
            data = self.model.infer(data)
            if self._cached_planet_id is None:
                self._cached_result = copy.deepcopy(data)
                self._cached_planet_id = [d.planet_id for d in data]
                
        mats = kgs.data_to_mats(data,data)
        if not self._disable_transforms:
            self.alter_mats(mats)
        kgs.mats_to_data(data,copy.deepcopy(data),mats)
            
        return data
    
@dataclass
class Fudger3(kgs.Model):
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])    
    sigma_offset: list = field(init=True, default_factory=lambda:[0.,0.])
    sigma_fudge_FGS = 1.
    sigma_fudge_AIRS_mean = 1.
    sigma_fudge_AIRS_var = 1.
    
    sigma_fudge_based_on_AIRS_var: list = field(init=True, default_factory=lambda:[0.,0.])
    fudge_based_on_AIRS_var = False
    
    adjust_based_on_u: list = field(init=True, default_factory=lambda:[0.,0.])
    do_adjust_based_on_u = False
    
    sigma_fudge_multi = 1.
    do_fudge_multi = False

    
    model: kgs.Model = field(init=True, default=None)
    
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
        if not self._cached_planet_id is None and [d.planet_id for d in data]==self._cached_planet_id:
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
    variance_fudge = 1.
    
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
        self.model.train(self._convert_data(train_data))
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


def baseline_model():
    model = Fudger2(model=ariel_gp.PredictionModel())
    model.model.starter_model.train(kgs.load_all_train_data())
    model.model.run_in_parallel = True
    return model


def combine_measurements(mu1, cov1, mu2, cov2):
    """
    Combine two independent Gaussian measurements of the same latent variable.

    Parameters
    ----------
    mu1, mu2 : (n,) cp.ndarray
        Mean vectors of the two measurements.
    cov1, cov2 : (n,n) cp.ndarray
        Covariance matrices of the two measurements.

    Returns
    -------
    mu : (n,) cp.ndarray
        Combined mean.
    cov : (n,n) cp.ndarray
        Combined covariance.
    """
    
    print(np.linalg.cond(cov1))
    print(np.linalg.cond(cov2))
    # precision matrices
    prec1 = np.linalg.inv(cov1)
    prec2 = np.linalg.inv(cov2)

    # combined covariance
    cov = np.linalg.inv(prec1 + prec2)

    # combined mean
    mu = cov @ (prec1 @ mu1 + prec2 @ mu2)

    return mu, cov