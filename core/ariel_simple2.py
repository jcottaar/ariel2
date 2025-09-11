import pandas as pd
import numpy as np
import scipy
import cupy as cp
import copy
from dataclasses import dataclass, field, fields
import kaggle_support as kgs
import ariel_numerics
import matplotlib.pyplot as plt
import batman
import ariel_transit

@dataclass
class SimpleModel(kgs.Model):
    
    # Configuration
    
    do_plots = False
    fixed_sigma: list = field(init=True, default_factory=lambda:[300e-6,500e-6]) # FGS, AIRS
    # output = a*pred + b, at spectrum level (not rp)
    #bias_a: list = field(init=True, default_factory=lambda:[0.000309, -0.0141]) # FGS, AIRS
    #bias_b: list = field(init=True, default_factory=lambda:[-1.93e-5, -1.88e-5]) # FGS, AIRS
    bias_a: list = field(init=True, default_factory=lambda:[0.,0.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.]) # FGS, AIRS
    
    # Configuration - step 1
    poly_order_step1 = 1
    t_steps = 100
    min_t0 = 2.5
    max_t0 = 5
    rp_vals: np.ndarray = field(init=True, default_factory=lambda:np.linspace(0,0.5,500))
    rp_init: list = field(init=True, default_factory=lambda:[0.05,0.05]) # FGS, AIRS
    u_init: list = field(init=True, default_factory=lambda:[[0.2,0.1],[0.2,0.1]]) # for FGS and AIRS
    force_kepler = False
    
    # Configuration - step 2
    do_step2 = True
    #weights: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    order_list: list = field(init=True, default_factory=lambda:[1,2,3]) 
    unlock_t0: bool = field(init=True, default=True)
    new_solver: bool = field(init=True, default=False)
    do_regularization:  bool = field(init=True, default=True)
    rescale_cost:  bool = field(init=True, default=True)
    
    # internal
    _targets = None # FGS, AIRS
    _times = None # FGS, AIRS
    _times_norm = None # FGS, AIRS
    
    # Set during inference
    # Transit parameters (batman)
    poly_order = None
    transit_param: list = field(init=True, default_factory = lambda:[ariel_transit.TransitParams(), ariel_transit.TransitParams()]) # FGS, AIRS

    # Other
    poly_vals = None # FGS, AIRS # ascending orders for Chebyshev, for FGS and AIRS
    
    # Diagnostics
    pred = None # FGS,AIRS
    cost_list = None 
    
    def _to_x(self):
        x = self.transit_param[0].to_x() + self.transit_param[1].to_x()[4:] + list(self.poly_vals[0]) + list(self.poly_vals[1])
        if self.unlock_t0:
            x.append(self.transit_param[1].t0 - self.transit_param[0].t0)
        else:
            assert self.transit_param[1].to_x()[:4]==x[:4]
        return list(x)
    
    def _from_x(self, x):
        x=list(x)
        len_xt = 7#len(self.transit_param[0].to_x())
        #print(len_xt)
        self.transit_param[0].from_x(x[:len_xt])
        self.transit_param[1].from_x(x[:4]+x[len_xt:2*len_xt-4])         
        cur_ind = 2*len_xt-4
        for ii in range(2):
            for jj in range(len(self.poly_vals[ii])):
                self.poly_vals[ii][jj] = x[cur_ind]
                cur_ind+=1
        if self.unlock_t0:
            self.transit_param[1].t0 = self.transit_param[0].t0 + x[cur_ind];cur_ind+=1;
        assert(cur_ind == len(x))
        if kgs.debugging_mode>=2:
            assert np.all( np.abs(np.array(self._to_x())-np.array(x))<=1e-10 )
            
    def _cov_mu(self):
        inds_self = [0,1,2,3,5,6,8,9,-1]
        inds_file = [0,2,3,4,5,7,6,8,1]
        
        cov_file, mu_file = kgs.dill_load(kgs.calibration_dir + 'transit_model_tuning9.pickle')    
        
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
        res = self.transit_param[sensor_id].light_curve(self._times[sensor_id])
        res *= np.polynomial.chebyshev.chebval(self._times_norm[sensor_id], self.poly_vals[sensor_id])        
        return res
        
    def _infer_single(self, data):
        # Load data
       # try:            
            data.transits[0].load_to_step(5, data, self.loaders)

            # Prepare stuff   
            for ii in range(2):
                self.transit_param[ii] = copy.deepcopy(data.transit_params)        
                self.transit_param[ii].force_kepler = self.force_kepler
                if abs(self.transit_param[ii].i-90)<0.1:
                    # If inc is close to 90 degrees, we can't get out of it due to the quadratic shape
                    self.transit_param[ii].i = 89.9
                self.transit_param[ii].Rp = self.rp_init[ii]
                self.transit_param[ii].u = self.u_init[ii]
                #xx=self.transit_param[ii].to_x()
                #xx[2]=0
                #self.transit_param[ii].from_x(xx)
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

            # Step 1
            for ii in range(2):
                self.transit_param[ii].t0 = np.max(self._times[ii])/2
                target_cp = cp.array(self._targets[ii])         
                base_curve = self._light_curve(ii)
                i_done=0
                while np.min(base_curve)>0.999:
                    if self.transit_param[ii].i>=90:
                        self.transit_param[ii].i-=0.01
                    else:
                        self.transit_param[ii].i+=0.01
                    base_curve = self._light_curve(ii)
                    i_done+=1
                    if i_done>=10000:
                        raise kgs.ArielException(0, 'Couldn''t find inc')
                #kgs.list_attrs(data)
                base_curve = np.concatenate([base_curve*0+1, base_curve, base_curve*0+1])
                base_curve = cp.array(base_curve)    
                cheb_mat = np.empty( (len(self._times[ii]), self.poly_order+1), dtype=np.float64)
                for jj in range(self.poly_order+1):
                    poly_vals = np.zeros( self.poly_order+1, dtype=np.float64)
                    poly_vals[jj]=1.
                    cheb_mat[:,jj] = np.polynomial.chebyshev.chebval(self._times_norm[ii], poly_vals)
                cheb_mat = cp.array(cheb_mat)
                t0_vals = np.round(np.linspace(0,len(self._times[ii])-1,self.t_steps)).astype(int)    
                t0_vals = t0_vals[(self._times[ii][t0_vals]>self.min_t0) & (self._times[ii][t0_vals]<self.max_t0)]
                res_mat = cp.empty((len(t0_vals), len(self.rp_vals)))
                rp_vals_cp = cp.array(self.rp_vals)
                for i_t, t0_ind in enumerate(t0_vals):
                    start_ind = t0_ind + len(self._times[ii])//2
                    this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]
                    this_res_mat = []
                    # for i_rp, rp in enumerate(self.rp_vals):                    
                    #     design_mat = cheb_mat#*(1-(rp/self.rp[ii])**2 *(1-this_curve[:,None]))
                    #     res_mat[i_t, i_rp] = cp.linalg.lstsq(design_mat, target_cp / (1-(rp/self.rp[ii])**2 *(1-this_curve[:])))[1][0]
                    res_mat[i_t, :] = cp.linalg.lstsq(cheb_mat, target_cp[:,None] / (1-(rp_vals_cp[None,:]/self.transit_param[ii].Rp)**2 *(1-this_curve[:,None])), rcond=None)[1]
                min_index = cp.unravel_index(cp.argmin(res_mat), res_mat.shape)           
                start_ind = t0_vals[min_index[0].get()] + len(self._times[ii])//2     
                this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]            
                this_res_mat = []                 
                design_mat = cheb_mat*(1-(self.rp_vals[min_index[1].get()]/self.transit_param[ii].Rp)**2 *(1-this_curve[:,None]))
                self.poly_vals[ii] = cp.linalg.lstsq(design_mat, target_cp, rcond=None)[0].get()            
                self.transit_param[ii].t0 = np.max(self._times[ii]) - self._times[ii][t0_vals[min_index[0].get()]]
                self.transit_param[ii].Rp = self.rp_vals[min_index[1].get()]
            self.transit_param[0].t0 = self.transit_param[1].t0

            step1=[]
            for ii in range(2):
                step1.append(self._light_curve(ii))
            # Step 2
            if self.new_solver:
                raise 'inactive'
                assert self.unlock_t0
                def residual(x):
                    self._from_x(x)
                    cost = 0
                    self.pred = [None]*2
                    residual = []
                    for ii in range(2):
                        self.pred[ii] = self._light_curve(ii) #* np.polynomial.chebyshev.chebval(self._times_norm[ii], self.poly_vals[ii])
                        residual.append(self.pred[ii]-self._targets[ii])
                    residual = np.concatenate(residual).flatten()
                    return residual
                def cost(x):
                    return kgs.rms(residual(x))
                self.cost_list=[]
                for o in self.order_list:
                    self.poly_order = o
                    for ii in range(2):
                        new_vals = np.zeros(self.poly_order+1)
                        inds_copy = min(len(new_vals), len(self.poly_vals[ii]))
                        new_vals[:inds_copy] = self.poly_vals[ii][:inds_copy]
                        self.poly_vals[ii] = new_vals
                    x0 = self._to_x()  
                    #res = scipy.optimize.minimize(cost,x0)
                    lb = -np.inf*np.ones(len(x0))
                    ub = np.inf*np.ones(len(x0))
                    lb[-1] = -0.1-0.001*o
                    ub[-1] = 0.1+0.001*o
                    lb[1]=-10-0.001*o
                    lb[2]=-0.5-0.001*o
                    ub[1]=10+0.001*o
                    ub[2]=0.5+0.001*o
                    res = scipy.optimize.least_squares(
                        fun=residual,
                        x0=x0,
                        bounds=(lb, ub),
                        method="trf",
                        loss="soft_l1",      # or "huber"
                        #f_scale=2.0,         # tunes outlier influence
                        #max_nfev=2000
                    )
                    def cost(x, do_plot = False):
                        self._from_x(x)
                        cost = 0
                        self.pred = [None]*2
                        for ii in range(2):
                            self.pred[ii] = self._light_curve(ii) #* np.polynomial.chebyshev.chebval(self._times_norm[ii], self.poly_vals[ii])
                            cost += np.sum( (self.pred[ii]-self._targets[ii])**2/noise_est[ii] )
                        return cost
                    #res = scipy.optimize.minimize(cost,res.x)        
                    self.cost_list.append(cost(res.x))
                #print(res.x)
                
            else:
                #print(noise_est)
                def cost(x, do_plot = False):
                    self._from_x(x)
                    cost = 0
                    self.pred = [None]*2
                    for ii in range(2):
                        self.pred[ii] = self._light_curve(ii) #* np.polynomial.chebyshev.chebval(self._times_norm[ii], self.poly_vals[ii])
                        cost += np.sum( (self.pred[ii]-self._targets[ii])**2 ) / noise_est[ii]**2
                        #cost += np.sqrt( np.mean((self.pred[ii]-self._targets[ii])**2))
                    if self.do_regularization:
                        xx = (x-reg_mu)[:,None]
                        cost += xx.T @ reg_prec @ xx
                    if self.rescale_cost:
                        cost = np.sqrt(cost) * np.mean(noise_est) / np.sqrt(1200)
                    else:
                        cost = cost * (np.mean(noise_est) / np.sqrt(1200))**2
                    return float(cost)
                self.cost_list=[]
                for o in self.order_list:
                    self.poly_order = o#self.poly_order_step2                    
                    for ii in range(2):
                        new_vals = np.zeros(self.poly_order+1)
                        inds_copy = min(len(new_vals), len(self.poly_vals[ii]))
                        new_vals[:inds_copy] = self.poly_vals[ii][:inds_copy]
                        self.poly_vals[ii] = new_vals
                    x0 = self._to_x()  
                    (reg_cov,reg_mu) = self._cov_mu()
                    reg_prec = np.linalg.inv(reg_cov)
                    lb = -np.inf*np.ones(len(x0))
                    ub = np.inf*np.ones(len(x0))                    
                    lb[1]=-5-0.001*o
                    ub[1]=0+0.001*o
                    lb[2]=-1-0.001*o                    
                    ub[2]=1+0.001*o
                    #lb[3]=-0.5-0.001*o                    
                    ub[3]=90
                    while True:
                        self._from_x(x0)
                        try:
                            res = scipy.optimize.minimize(cost,x0)                           
                        except:
                            self._from_x(x0)
                            res = scipy.optimize.minimize(cost,x0, bounds=zip(lb,ub))      
                        if res.x[3]>90:
                            res.x[3] = 90-res.x[3]
                            x0 = res.x
                        else:
                            break
                    # while res.x[3]>90:
                    #     res.x[3] = 90-res.x[3]
                    #     res = scipy.optimize.minimize(cost,res.x)               
                    self.cost_list.append(cost(res.x))


            
            
            # sanity checks: t0, ecc, noise ratio
            
            #kgs.sanity_check(lambda x:x, self.ecc, 'simple_ecc', 12, [-0.25,0.25])
            #print('sanity back')
            for ii in range(2):
                #noise_estimate = ariel_numerics.estimate_noise(self._targets[ii])
                residual = kgs.rms(self._targets[ii]-self.pred[ii])
                residual_filtered = ariel_numerics.estimate_noise(self._targets[ii]-self.pred[ii])
                ratio = residual/residual_filtered
                if ii==0:
                    data.diagnostics['simple_residual_diff_FGS'] = ratio
                    #print('FGS', ratio, residual/residual_filtered)
                    #kgs.sanity_check(lambda x:x, ratio, 'simple_residual_diff_FGS', 12, [0.9,1.3])
                else:
                    data.diagnostics['simple_residual_diff_AIRS'] = ratio
                    #print('AIRS', ratio, residual/residual_filtered)
                    #kgs.sanity_check(lambda x:x, ratio, 'simple_residual_diff_AIRS', 13, [0.9,1.3]) # up to ~3e-5 in training, but up to ~12e-5 in test
            #print(self.transit_param)
            
            if self.do_plots:            
                _,ax = plt.subplots(2,3,figsize=(15,10))
                for ii in range(2):
                    plt.sca(ax[0,ii])
                    plt.grid(True)
                    plt.xlabel('Time [h]')
                    plt.ylabel('Normalized intensity')
                    plt.plot(self._times[ii], self._targets[ii])
                    plt.plot(self._times[ii], step1[ii])
                    plt.plot(self._times[ii], self.pred[ii])
                    plt.legend(('Measured', 'After step 1', 'After step 2'))
                    plt.sca(ax[1,ii])
                    plt.plot(self._targets[ii]-self.pred[ii])
                plt.sca(ax[0,2])
                plt.plot(self.order_list[1:], self.cost_list[1:])
                plt.grid(True)
                plt.xlabel('Poly order')
                plt.ylabel('Cost')
                plt.pause(0.001)
                print( data.diagnostics['simple_residual_diff_FGS'], data.diagnostics['simple_residual_diff_AIRS'])


            # Report resutls
            data.spectrum = np.concatenate([
                [(1+self.bias_a[0])*(self.transit_param[0].Rp**2)+self.bias_b[0]], 
                (1+self.bias_a[1])*(self.transit_param[1].Rp**2)*np.ones(282)+self.bias_b[1]])
            sigma = np.concatenate([[self.fixed_sigma[0]], self.fixed_sigma[1]*np.ones(282)])
            data.spectrum_cov = np.diag(sigma**2)
            data.diagnostics['t0'] = self.transit_param[0].t0
            midpoint = np.min(self.pred[0])/2+np.max(self.pred[0])/2
            data.diagnostics['t_ingress'] = self._times[0][np.argwhere(self.pred[0]<midpoint)[0,0]]
            data.diagnostics['t_egress'] = self._times[0][np.argwhere(self.pred[0]<midpoint)[-1,0]]
            data.diagnostics['transit_params'] = copy.deepcopy(self.transit_param)
            #data.diagnostics['poly_vals'] = copy.deepcopy(self.poly_vals)
            data.check_constraints()
            
            return data
        # except Exception as err:
        #     import traceback
        #     import sys
        #     exc_type, exc_value, exc_tb = sys.exc_info()
        #     tb = traceback.extract_tb(exc_tb)
        #     filename, lineno, funcname, text = tb[-1]
        #     raise kgs.ArielException((lineno-116)/10,text)


@dataclass
class SimpleModelChainer(kgs.Model):
    model: SimpleModel = field(init=True, default_factory=SimpleModel)
    
    _train_data = None
    
    ok_threshold = 1.02
    chop_threshold = 1.05
    n_alternative_params = 10
    improvement_threshold_chopping = 2
    chop_amount = 150
    
    def _train(self,train_data):
        self.model.train(train_data)
        self._train_data = train_data
        
    def _infer_single(self,data):
        def try_one(data, early_stop):
            data.transits[0].load_to_step(5, data, self.model.loaders)
            data_results = []
            score_results = []
            for ii in range(self.n_alternative_params+2):
                model = copy.deepcopy(self.model)
                model.new_solver = False
                dat = copy.deepcopy(data)
                if ii==1:
                    if early_stop:
                        print(f'New solver/alternative transit parameters fallback for planet id {data.planet_id}')
                    continue
                    model.new_solver = True
                elif ii>2:
                    model.new_solver = False
                    dat.transit_params = self._train_data[ii-2].transit_params
                dat = self.model.infer([dat])[0]
                data_results.append(dat)
                score_results.append(max(dat.diagnostics['simple_residual_diff_AIRS'],dat.diagnostics['simple_residual_diff_FGS']))
                if ii==0 and early_stop and score_results[-1]<self.ok_threshold:
                    return data_results[-1], score_results[-1]
            best_ind = np.argmin(score_results)
            return data_results[best_ind], score_results[best_ind]
        
        [result, score] = try_one(data, True)
        result.diagnostics['chopped'] = False
        
        if score>self.chop_threshold:
            print('Attempting chop')
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
                    print(f'Chop used on planet id {result.planet_id}')
                    result.diagnostics['chopped'] = True
                    result = alt_result
                    score = alt_score
                    break
        
        result.diagnostics['suspicious_AIRS'] = result.diagnostics['simple_residual_diff_AIRS']>self.ok_threshold
        result.diagnostics['suspicious_FGS'] = result.diagnostics['simple_residual_diff_FGS']>self.ok_threshold
        
        #kgs.sanity_check(lambda x:x, result.diagnostics['transit_params'][1].t0, 'simple_t0_AIRS', 11, [2.5, 5])
        #kgs.sanity_check(lambda x:x, result.diagnostics['transit_params'][1].t0-result.diagnostics['transit_params'][0].t0, 'simple_t0_diff', 11, [-0.1,0.1])
        kgs.sanity_check(lambda x:x, result.diagnostics['simple_residual_diff_FGS'], 'simple_residual_diff_FGS', 12, [0.95,1.1])
        kgs.sanity_check(lambda x:x, result.diagnostics['simple_residual_diff_AIRS'], 'simple_residual_diff_AIRS', 13, [0.95,1.15])
        
        return result

                
