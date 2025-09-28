'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This class defines a simple 'starter model', used to find a good starting point for the full Bayesian solver.
'''

import numpy as np
import scipy
import cupy as cp
import copy
from dataclasses import dataclass, field
import kaggle_support as kgs
import ariel_numerics
import matplotlib.pyplot as plt
import ariel_transit

@dataclass
class SimpleModel(kgs.Model):
    # Makes a prediction and can be used as a submission model, but mainly designed to find a good starting point transit parameters.
    # Only deals with FGS and mean over wavelengths of AIRS.
    
    # Configuration
    fixed_sigma: list = field(init=True, default_factory=lambda:[300e-6,500e-6]) # FGS, AIRS; only relevant when using as submission model
    
    # Configuration - step 1 (grid search)
    poly_order_step1 = 1 # gain drift order
    t_steps = 100 # time steps to consider
    min_t0 = 2.5 # minimum mid-transit time (in hours)
    max_t0 = 5 # maximum mid-transit time (in hours)
    rp_vals: np.ndarray = field(init=True, default_factory=lambda:np.linspace(0,0.5,500)) # Rp values (transit depth) to consider
    rp_init: list = field(init=True, default_factory=lambda:[0.05,0.05]) # starting point for Rp (FGS, AIRS)
    u_init: list = field(init=True, default_factory=lambda:[[0.2,0.1],[0.2,0.1]]) # starting point for limb darkening parameters (FGS, AIRS)
    
    # Configuration - step 2 (scipy.optimize)
    do_step2 = True # do step 2 at all?
    order_list: list = field(init=True, default_factory=lambda:[1,2,3]) # do step 2 once for each order, in sequence
    unlock_t0: bool = field(init=True, default=True) # allow different t0 values for FGS and AIRS
    do_regularization:  bool = field(init=True, default=True) # apply a penalty for unrealistic transit parameters
    reg_file = kgs.calibration_dir + 'transit_model_tuning9.pickle' # file containing the regularization prior for the transit parameteres
    rescale_cost:  bool = field(init=True, default=True) # use alternative cost function
    
    # internal
    _targets = None # FGS, AIRS
    _times = None # FGS, AIRS
    _times_norm = None # FGS, AIRS
    
    # Set during inference
    poly_order = None # final polynomial order used for gain drift
    transit_param: list = field(init=True, default_factory = lambda:[ariel_transit.TransitParams(), ariel_transit.TransitParams()]) # FGS, AIRS

    # Other
    poly_vals = None # ascending orders for Chebyshev, for FGS and AIRS
    
    # Diagnostics
    pred = None 
    cost_list = None 
    
    def _to_x(self):
        # Convert own parameters to vector
        x = self.transit_param[0].to_x() + self.transit_param[1].to_x()[4:] + list(self.poly_vals[0]) + list(self.poly_vals[1])
        if self.unlock_t0:
            x.append(self.transit_param[1].t0 - self.transit_param[0].t0)
        else:
            assert self.transit_param[1].to_x()[:4]==x[:4]
        return list(x)    
    def _from_x(self, x):
        # Inverse of above
        x=list(x)
        len_xt = 7
        self.transit_param[0].from_x(x[:len_xt])
        self.transit_param[1].from_x(x[:4]+x[len_xt:2*len_xt-4])         
        cur_ind = 2*len_xt-4
        for ii in range(2):
            for jj in range(len(self.poly_vals[ii])):
                self.poly_vals[ii][jj] = x[cur_ind]
                cur_ind+=1
        if self.unlock_t0:
            self.transit_param[1].t0 = self.transit_param[0].t0 + x[cur_ind];cur_ind+=1
        assert(cur_ind == len(x))
        if kgs.debugging_mode>=2:
            assert np.all( np.abs(np.array(self._to_x())-np.array(x))<=1e-10 )
            
    def _cov_mu(self):
        # Find regularization prior. This file follows a different parameter order than ariel_gp.py, so we need to do some shuffling.
        inds_self = [0,1,2,3,5,6,8,9,-1]
        inds_file = [0,2,3,4,5,7,6,8,1]
        
        cov_file, mu_file = kgs.dill_load(self.reg_file)    
        
        cov = np.diag(1e4*np.ones(len(self._to_x())))
        mu = np.zeros(len(self._to_x()))
        
        cov[np.ix_(inds_self, inds_self)] = cov_file[np.ix_(inds_file, inds_file)]
        mu[inds_self] = mu_file[inds_file]
        
        # Deal with FGS t0 - AIRS t0
        mu[-1] = mu_file[1]-mu_file[0]
        cov[-1,:] = 0
        cov[:,-1] = 0
        cov[-1,-1] = cov_file[1, 1] + cov_file[0, 0] - 2.0 * cov_file[0, 1]
        
        return cov, mu
        
    
    def _light_curve(self, sensor_id):     
        # Compute light curve for a given sensor
        res = self.transit_param[sensor_id].light_curve(self._times[sensor_id])
        res *= np.polynomial.chebyshev.chebval(self._times_norm[sensor_id], self.poly_vals[sensor_id])        
        return res
        
    def _infer_single(self, data):
        
        # Load data 
        data.transits[0].load_to_step(5, data, self.loaders)

        # Prepare stuff   
        for ii in range(2):
            self.transit_param[ii] = copy.deepcopy(data.transit_params)        
            if abs(self.transit_param[ii].i-90)<0.1:
                # If inc is close to 90 degrees, we can't get out of it due to the quadratic shape
                self.transit_param[ii].i = 89.9
            self.transit_param[ii].Rp = self.rp_init[ii]
            self.transit_param[ii].u = self.u_init[ii]
        self.poly_order = self.poly_order_step1
        self.poly_vals = [np.zeros(self.poly_order+1), np.zeros(self.poly_order+1)]                
        self._targets = [None, None]
        self._times = [None, None]
        self._times_norm = [None, None]
        noise_est = [None,None]
        for ii in range(2):
            self.poly_vals[ii][0] = 1
            self._targets[ii] = cp.mean(data.transits[0].data[ii].data, axis=1)
            self._targets[ii] /= cp.mean(self._targets[ii])
            self._targets[ii] = self._targets[ii].get()
            self._times[ii] = data.transits[0].data[ii].times.get()/3600
            t=self._times[ii]
            self._times_norm[ii] = 2*(t - t.min())/(t.max() - t.min()) - 1
            noise_est[ii] = ariel_numerics.estimate_noise_cp(self._targets[ii])

        # Step 1: grid search over t0 and Rp
        for ii in range(2):
            self.transit_param[ii].t0 = np.max(self._times[ii])/2
            target_cp = cp.array(self._targets[ii])         
            base_curve = self._light_curve(ii)
            i_done=0
            
            # Make sure there is a transit at all
            while np.min(base_curve)>0.999:
                if self.transit_param[ii].i>=90:
                    self.transit_param[ii].i-=0.01
                else:
                    self.transit_param[ii].i+=0.01
                base_curve = self._light_curve(ii)
                i_done+=1
                if i_done>=10000:
                    raise kgs.ArielException(0, 'Couldn''t find inc')
                    
            # Compute light curve (we do it once and then manipulate it during the grid search)
            base_curve = np.concatenate([base_curve*0+1, base_curve, base_curve*0+1])
            base_curve = cp.array(base_curve)    
            
            # Set up gain drift marix
            cheb_mat = np.empty( (len(self._times[ii]), self.poly_order+1), dtype=np.float64)
            for jj in range(self.poly_order+1):
                poly_vals = np.zeros( self.poly_order+1, dtype=np.float64)
                poly_vals[jj]=1.
                cheb_mat[:,jj] = np.polynomial.chebyshev.chebval(self._times_norm[ii], poly_vals)
            cheb_mat = cp.array(cheb_mat)
            
            # Grid search
            t0_vals = np.round(np.linspace(0,len(self._times[ii])-1,self.t_steps)).astype(int)    
            t0_vals = t0_vals[(self._times[ii][t0_vals]>self.min_t0) & (self._times[ii][t0_vals]<self.max_t0)]
            res_mat = cp.empty((len(t0_vals), len(self.rp_vals)))
            rp_vals_cp = cp.array(self.rp_vals)
            for i_t, t0_ind in enumerate(t0_vals):
                start_ind = t0_ind + len(self._times[ii])//2
                this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]
                this_res_mat = []
                res_mat[i_t, :] = cp.linalg.lstsq(cheb_mat, target_cp[:,None] / (1-(rp_vals_cp[None,:]/self.transit_param[ii].Rp)**2 *(1-this_curve[:,None])), rcond=None)[1]
            min_index = cp.unravel_index(cp.argmin(res_mat), res_mat.shape)           
            start_ind = t0_vals[min_index[0].get()] + len(self._times[ii])//2     
            
            # Store some stuff
            this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]            
            this_res_mat = []                 
            design_mat = cheb_mat*(1-(self.rp_vals[min_index[1].get()]/self.transit_param[ii].Rp)**2 *(1-this_curve[:,None]))
            self.poly_vals[ii] = cp.linalg.lstsq(design_mat, target_cp, rcond=None)[0].get()            
            self.transit_param[ii].t0 = np.max(self._times[ii]) - self._times[ii][t0_vals[min_index[0].get()]]
            self.transit_param[ii].Rp = self.rp_vals[min_index[1].get()]
        self.transit_param[0].t0 = self.transit_param[1].t0

        # Step 2: gradient descent
        def cost(x, do_plot = False):
            # Cost function for gradient descent
            self._from_x(x)
            cost = 0
            self.pred = [None]*2
            for ii in range(2):
                self.pred[ii] = self._light_curve(ii) 
                cost += np.sum( (self.pred[ii]-self._targets[ii])**2 ) / noise_est[ii]**2
            if self.do_regularization:
                # Apply penalty to transit parameters
                xx = (x-reg_mu)[:,None]
                cost += xx.T @ reg_prec @ xx
            if self.rescale_cost:
                # Not sure why this is needed, but it is
                cost = np.sqrt(cost) * np.mean(noise_est) / np.sqrt(1200)
            else:
                cost = cost * (np.mean(noise_est) / np.sqrt(1200))**2
            return float(cost)
        
        self.cost_list=[]
        for o in self.order_list:
            
            # Move up in polynomial order one by one, to prevent polynomial from eating transit content
            self.poly_order = o
            
            # Take over previous polynomial values
            for ii in range(2):
                new_vals = np.zeros(self.poly_order+1)
                inds_copy = min(len(new_vals), len(self.poly_vals[ii]))
                new_vals[:inds_copy] = self.poly_vals[ii][:inds_copy]
                self.poly_vals[ii] = new_vals
                
            # Prepare and set bounds
            x0 = self._to_x()  
            (reg_cov,reg_mu) = self._cov_mu()
            reg_prec = np.linalg.inv(reg_cov)
            lb = -np.inf*np.ones(len(x0))
            ub = np.inf*np.ones(len(x0))                    
            lb[1]=-5-0.001*o
            ub[1]=0+0.001*o
            lb[2]=-1-0.001*o                    
            ub[2]=1+0.001*o                
            ub[3]=90
            
            # Optimize
            while True:
                self._from_x(x0)
                try:
                    # First try without bounds
                    res = scipy.optimize.minimize(cost,x0)                           
                except:
                    # If we end up in weird spots, use bounds
                    self._from_x(x0)
                    res = scipy.optimize.minimize(cost,x0, bounds=zip(lb,ub))      
                if res.x[3]>90:
                    # If we ended up above 90 degrees inclination, invert it and try again
                    res.x[3] = 90-res.x[3]
                    x0 = res.x
                else:
                    break           
            self.cost_list.append(cost(res.x))

        # Compute residual for diagnostics
        for ii in range(2):
            residual = kgs.rms(self._targets[ii]-self.pred[ii])
            residual_filtered = ariel_numerics.estimate_noise(self._targets[ii]-self.pred[ii])
            ratio = residual/residual_filtered
            if ii==0:
                data.diagnostics['simple_residual_diff_FGS'] = ratio
            else:
                data.diagnostics['simple_residual_diff_AIRS'] = ratio
        

        # Report results
        data.spectrum = np.concatenate([
            [(self.transit_param[0].Rp**2)], 
            (self.transit_param[1].Rp**2)*np.ones(282)])
        sigma = np.concatenate([[self.fixed_sigma[0]], self.fixed_sigma[1]*np.ones(282)])
        data.spectrum_cov = np.diag(sigma**2)
        midpoint = np.min(self.pred[0])/2+np.max(self.pred[0])/2
        data.diagnostics['transit_params'] = copy.deepcopy(self.transit_param)
        data.check_constraints()

        return data


@dataclass
class SimpleModelChainer(kgs.Model):
    # This class uses the model above, but tries some fallbacks if the residual is too high.
    
    model: SimpleModel = field(init=True, default_factory=SimpleModel) # underlying model
    
    _train_data = None # store some train data internally
    
    ok_threshold = 1.02 # if residual is below this threshold, we're done
    
    n_alternative_params = 10 # how many alternative transit parameters to try
    
    chop_threshold = 1.05 # if residual is above this threshold, consider chopping off the start or end
    improvement_threshold_chopping = 2 # residual must improve by this much to allow chopping
    chop_amount = 150 # how many frames to chop (same for FGS and AIRS)
    
    def _train(self,train_data):
        self.model.train(train_data) # train underlying model
        self._train_data = train_data # store train data (to use the transit parameters later)
        
    def _infer_single(self,data):
        
        def try_one(data, early_stop):
            
            # This is where we actually end up loading the data. Ensure only one process does this at a time to efficiently use GPU.
            with kgs.gpu_semaphores[kgs.my_gpu_id]:
                data.transits[0].load_to_step(5, data, self.loaders)
                kgs.clear_gpu()

            # Apply the underlying model and use fallback if not good enough
            data_results = []
            score_results = []
            for ii in range(self.n_alternative_params+1):
                model = copy.deepcopy(self.model) 
                dat = copy.deepcopy(data)
                if ii>1:
                    # Try transit parameters for a different planet as starting point
                    old_Ts = dat.transit_params.Ts
                    dat.transit_params = self._train_data[ii-2].transit_params
                    dat.transit_params.Ts = old_Ts
                dat = self.model.infer([dat])[0] # apply underlyig model
                data_results.append(dat)
                score_results.append(max(dat.diagnostics['simple_residual_diff_AIRS'],dat.diagnostics['simple_residual_diff_FGS']))
                if ii==0 and early_stop and score_results[-1]<self.ok_threshold:
                    # If the first result is good enough, stop here
                    return data_results[-1], score_results[-1]
                
            # Pick best result
            best_ind = np.argmin(score_results)
            return data_results[best_ind], score_results[best_ind]
        
        [result, score] = try_one(data, True)
        result.diagnostics['chopped'] = False
        
        if score>self.chop_threshold:
            # If score is still not good enough, try chopping off start or end of signal, and see if this improves matters
            for chop_right in [False,True]:
                if chop_right:
                    slic = slice(None,-self.chop_amount)
                else:
                    slic = slice(self.chop_amount,None)
                d = copy.deepcopy(data)
                d.transits[0].data[1].data = d.transits[0].data[1].data[slic,...]
                d.transits[0].data[1].times = d.transits[0].data[1].times[slic,...]
                d.transits[0].data[1].time_intervals = d.transits[0].data[1].time_intervals[slic,...]
                [alt_result, alt_score] = try_one(d, True)
                if np.abs(alt_score-1)<0.5*(np.abs(score-1)):
                    result.diagnostics['chopped'] = True
                    result = alt_result
                    score = alt_score
                    break
        
        # Diagnostics and sanity checks
        result.diagnostics['suspicious_AIRS'] = result.diagnostics['simple_residual_diff_AIRS']>self.ok_threshold
        result.diagnostics['suspicious_FGS'] = result.diagnostics['simple_residual_diff_FGS']>self.ok_threshold
        kgs.sanity_check(lambda x:x, result.diagnostics['simple_residual_diff_FGS'], 'simple_residual_diff_FGS', 12, [0.95,1.1])
        kgs.sanity_check(lambda x:x, result.diagnostics['simple_residual_diff_AIRS'], 'simple_residual_diff_AIRS', 13, [0.95,1.15])
        
        return result

                
