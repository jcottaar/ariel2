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

@dataclass
class SimpleModel(kgs.Model):
    
    # Configuration
    supersample_factor = 1
    max_err = 1.
    do_plots = False
    fixed_sigma: list = field(init=True, default_factory=lambda:[300e-6,500e-6]) # FGS, AIRS
    # output = a*pred + b, at spectrum level (not rp)
    bias_a: list = field(init=True, default_factory=lambda:[0.001323, -0.01358]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[-3.033e-5, -2.2346e-5]) # FGS, AIRS
    
    # Configuration - step 1
    poly_order_step1 = 1
    t_steps = 100
    rp_vals: np.ndarray = field(init=True, default_factory=lambda:np.linspace(0,0.5,500))
    rp_init: list = field(init=True, default_factory=lambda:[0.05,0.05]) # FGS, AIRS
    u_init: list = field(init=True, default_factory=lambda:[[0.2,0.1],[0.2,0.1]]) # for FGS and AIRS
    
    # Configuration - step 2
    do_step2 = True
    fit_ecc = False
    weights: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    use_correction_factor = False
    order_list: list = field(init=True, default_factory=lambda:[0,1,2,3,4,5]) # FGS, AIRS
    
    # internal
    _targets = None # FGS, AIRS
    _times = None # FGS, AIRS
    _times_norm = None # FGS, AIRS
    
    # Set during inference
    # Transit parameters (batman)
    poly_order = None
    t0 = None # in seconds
    per = None # in seconds
    rp = None # FGS, AIRS
    a = None
    inc = None
    ecc = None
    w = None
    u = None
    limb_dark = 'quadratic'
    correction_factor = None
    # Other
    poly_vals = None # FGS, AIRS # ascending orders for Chebyshev, for FGS and AIRS
    
    # Diagnostics
    pred = None # FGS,AIRS
    cost_list = None 
    
    def _to_x(self):
        x = [self.t0/3600, self.per/24/3600, self.a, self.inc, self.ecc, self.w]
        for ii in range(2):
            #x = x+[self.rp[ii]]+list(self.u[ii])+list(self.poly_vals[ii])
            x = x+[self.rp[ii]]+[self.correction_factor[ii]]+list(self.u[ii])+list(self.poly_vals[ii])
        return x
    
    def _from_x(self, x):
        self.t0 = x[0]*3600
        self.per = x[1]*24*3600
        self.a = x[2]
        self.inc = x[3]
        self.ecc = x[4]
        self.w = x[5]        

        index = 6
        for ii in range(2):
            self.rp[ii] = x[index]
            index += 1
            self.correction_factor[ii] = x[index]
            index += 1
            for jj in range(len(self.u[ii])):
                self.u[ii][jj] = x[index]
                index += 1
            for jj in range(len(self.poly_vals[ii])):
                self.poly_vals[ii][jj] = x[index]
                index += 1
        if kgs.debugging_mode>=2:
            assert np.all( np.abs(self._to_x()-x)<=1e-10 )
            
    
    def _light_curve(self, sensor_id):
        params = batman.TransitParams()
        params.t0 = self.t0
        params.per = self.per
        params.rp = self.rp[sensor_id]
        params.a = self.a
        params.inc = self.inc
        if self.fit_ecc:
            params.ecc = self.ecc
        else:
            params.ecc = 0
        params.w = self.w   
        params.limb_dark = self.limb_dark
        params.u = self.u[sensor_id] 
        model=batman.TransitModel(params, self._times[sensor_id], exp_time=(self._times[sensor_id][1]-self._times[sensor_id][0]), supersample_factor=self.supersample_factor, max_err=self.max_err)
        res = model.light_curve(params)
        if self.use_correction_factor:
            res = 1-self.correction_factor[sensor_id]*(1-res)
        res *= np.polynomial.chebyshev.chebval(self._times_norm[sensor_id], self.poly_vals[sensor_id])        
        return res
        
    def _infer_single(self, data):
        # Load data
       # try:            
            data.transits[0].load_to_step(5, data, self.loaders)

            # Prepare stuff       
            self.poly_order = self.poly_order_step1
            self.per = data.P*24*3600
            self.t0 = 3.5*3600
            self.a = data.sma
            self.inc = data.i
            if abs(self.inc-90)<0.1:
                # If inc is close to 90 degrees, we can't get out of it due to the quadratic shape
                self.inc = 89.9
            self.ecc = data.e
            self.w = 90
            self.rp = copy.deepcopy(self.rp_init)
            self.u = copy.deepcopy(self.u_init)
            self.correction_factor = [1.,1.];
            self.poly_vals = [np.zeros(self.poly_order+1), np.zeros(self.poly_order+1)]                
            self._targets = [None, None]
            self._times = [None, None]
            self._times_norm = [None, None]
            for ii in range(2):
                self.poly_vals[ii][0] = 1
                self._targets[ii] = cp.mean(data.transits[0].data[ii].data, axis=1)
                self._targets[ii] /= cp.mean(self._targets[ii])
                self._targets[ii] = self._targets[ii].get()
                self._times[ii] = data.transits[0].data[ii].times.get()
                t=self._times[ii]
                self._times_norm[ii] = 2*(t - t.min())/(t.max() - t.min()) - 1

            # Step 1
            for ii in range(2):
                self.t0 = np.max(self._times[ii])/2
                target_cp = cp.array(self._targets[ii])            
                base_curve = self._light_curve(ii)
                i_done=0
                while np.min(base_curve)>0.999:
                    if self.inc>=90:
                        self.inc-=0.01
                    else:
                        self.inc+=0.01
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
                t0_vals = np.round(np.linspace(0,len(self._times[ii]),self.t_steps)).astype(int)
                res_mat = cp.empty((len(t0_vals), len(self.rp_vals)))
                rp_vals_cp = cp.array(self.rp_vals)
                for i_t, t0_ind in enumerate(t0_vals):
                    start_ind = t0_ind + len(self._times[ii])//2
                    this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]
                    this_res_mat = []
                    # for i_rp, rp in enumerate(self.rp_vals):                    
                    #     design_mat = cheb_mat#*(1-(rp/self.rp[ii])**2 *(1-this_curve[:,None]))
                    #     res_mat[i_t, i_rp] = cp.linalg.lstsq(design_mat, target_cp / (1-(rp/self.rp[ii])**2 *(1-this_curve[:])))[1][0]
                    res_mat[i_t, :] = cp.linalg.lstsq(cheb_mat, target_cp[:,None] / (1-(rp_vals_cp[None,:]/self.rp[ii])**2 *(1-this_curve[:,None])), rcond=None)[1]
                min_index = cp.unravel_index(cp.argmin(res_mat), res_mat.shape)           
                start_ind = t0_vals[min_index[0].get()] + len(self._times[ii])//2            
                this_curve = base_curve[start_ind:start_ind+len(self._times[ii])]            
                this_res_mat = []                 
                design_mat = cheb_mat*(1-(self.rp_vals[min_index[1].get()]/self.rp[ii])**2 *(1-this_curve[:,None]))
                self.poly_vals[ii] = cp.linalg.lstsq(design_mat, target_cp, rcond=None)[0].get()            
                self.t0 = np.max(self._times[ii]) - self._times[ii][t0_vals[min_index[0].get()]]
                self.rp[ii] = self.rp_vals[min_index[1].get()]

            step1=[]
            for ii in range(2):
                step1.append(self._light_curve(ii))

            # Step 2
            def cost(x, do_plot = False):
                self._from_x(x)
                cost = 0
                self.pred = [None]*2
                for ii in range(2):
                    self.pred[ii] = self._light_curve(ii) #* np.polynomial.chebyshev.chebval(self._times_norm[ii], self.poly_vals[ii])
                    cost += self.weights[ii]*np.sqrt(np.mean( (self.pred[ii]-self._targets[ii])**2 ))
                return cost
            self.cost_list=[]
            for o in self.order_list:
                self.poly_order = o#self.poly_order_step2
                for ii in range(2):
                    new_vals = np.zeros(self.poly_order+1)
                    inds_copy = min(len(new_vals), len(self.poly_vals[ii]))
                    new_vals[:inds_copy] = self.poly_vals[ii][:inds_copy]
                    self.poly_vals[ii] = new_vals
                x0 = self._to_x()  
                res = scipy.optimize.minimize(cost,x0)
                self.cost_list.append(cost(res.x))


            if self.do_plots:            
                _,ax = plt.subplots(2,3,figsize=(15,10))
                for ii in range(2):
                    plt.sca(ax[0,ii])
                    plt.grid(True)
                    plt.xlabel('Time [h]')
                    plt.ylabel('Normalized intensity')
                    plt.plot(self._times[ii]/3600, self._targets[ii])
                    plt.plot(self._times[ii]/3600, step1[ii])
                    plt.plot(self._times[ii]/3600, self.pred[ii])
                    plt.legend(('Measured', 'After step 1', 'After step 2'))
                    plt.sca(ax[1,ii])
                    plt.plot(self._targets[ii]-self.pred[ii])
                plt.sca(ax[0,2])
                plt.plot(self.order_list[1:], self.cost_list[1:])
                plt.grid(True)
                plt.xlabel('Poly order')
                plt.ylabel('Cost')
                plt.pause(0.001)

            self._times = None
            self._times_norm = None

            # sanity checks: t0, ecc, noise ratio
            kgs.sanity_check(lambda x:x, self.t0, 'simple_t0', 11, [2.5*3600, 5*3600])
            #kgs.sanity_check(lambda x:x, self.ecc, 'simple_ecc', 12, [-0.25,0.25])
            for ii in range(2):
                #noise_estimate = ariel_numerics.estimate_noise(self._targets[ii])
                residual = kgs.rms(self._targets[ii]-self.pred[ii])
                residual_filtered = ariel_numerics.estimate_noise(self._targets[ii]-self.pred[ii])
                #print(noise_estimate, residual)
                ratio = residual-residual_filtered
                if ii==0:
                    data.diagnostics['simple_residual_diff_FGS'] = ratio
                    kgs.sanity_check(lambda x:x, ratio, 'simple_residual_diff_FGS', 12, [-np.inf,2.])
                else:
                    data.diagnostics['simple_residual_diff_AIRS'] = ratio
                    kgs.sanity_check(lambda x:x, ratio, 'simple_residual_diff_AIRS', 13, [-np.inf,2.])


            # Report resutls
            data.spectrum = np.concatenate([
                [(1+self.bias_a[0])*(self.rp[0]**2*self.correction_factor[0])+self.bias_b[0]], 
                (1+self.bias_a[1])*(self.rp[1]**2*self.correction_factor[1])*np.ones(282)+self.bias_b[1]])
            sigma = np.concatenate([[self.fixed_sigma[0]], self.fixed_sigma[1]*np.ones(282)])
            data.spectrum_cov = np.diag(sigma**2)
            data.check_constraints()
            return data
        # except Exception as err:
        #     import traceback
        #     import sys
        #     exc_type, exc_value, exc_tb = sys.exc_info()
        #     tb = traceback.extract_tb(exc_tb)
        #     filename, lineno, funcname, text = tb[-1]
        #     raise kgs.ArielException((lineno-116)/10,text)