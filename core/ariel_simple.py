import pandas as pd
import numpy as np
import scipy
import cupy as cp
import copy
from dataclasses import dataclass, field, fields
import kaggle_support as kgs
import matplotlib.pyplot as plt
import batman

@dataclass
class SimpleModel(kgs.Model):
    
    # Configuration
    supersample_factor = 1
    do_plots = False
    fixed_sigma: list = field(init=True, default_factory=lambda:[0.001,0.001]) # FGS, AIRS
    
    # Configuration - step 1
    poly_order_step1 = 1
    t_steps = 100
    rp_vals: np.ndarray = field(init=True, default_factory=lambda:np.linspace(0,0.5,500))
    
    # Configuration - step 2
    do_step2 = True
    order_list: list = field(init=True, default_factory=lambda:[0,1,2,3,4,5]) # FGS, AIRS
    
    # internal
    _times = None # FGS, AIRS
    _times_norm = None # FGS, AIRS
    
    # Trained
    # Transit parameters (batman)
    poly_order = None
    t0 = None # in seconds
    per = None # in seconds
    rp: list = field(init=True, default_factory=lambda:[0.05,0.05]) # FGS, AIRS
    a = None
    inc = None
    ecc = None
    w = 90
    u: list = field(init=True, default_factory=lambda:[[0.2,0.1],[0.2,0.1]]) # for FGS and AIRS
    limb_dark = 'quadratic'
    # Other
    poly_vals = None # FGS, AIRS # ascending orders for Chebyshev, for FGS and AIRS
    
    # Diagnostics
    pred = None # FGS,AIRS
    cost_list = None 
    
    def _to_x(self):
        x = [self.t0, self.per, self.a, self.inc, self.ecc, self.w]
        for ii in range(2):
            x = x+[self.rp[ii]]+list(self.u[ii])+list(self.poly_vals[ii])
        return x
    
    def _from_x(self, x):
        self.t0 = x[0]
        self.per = x[1]
        self.a = x[2]
        self.inc = x[3]
        self.ecc = x[4]
        self.w = x[5]

        index = 6
        for ii in range(2):
            self.rp[ii] = x[index]
            index += 1
            for jj in range(len(self.u[ii])):
                self.u[ii][jj] = x[index]
                index += 1
            for jj in range(len(self.poly_vals[ii])):
                self.poly_vals[ii][jj] = x[index]
                index += 1
        if kgs.debugging_mode>=2:
            assert np.all(self._to_x()==x)
            
    
    def _light_curve(self, sensor_id):
        params = batman.TransitParams()
        params.t0 = self.t0
        params.per = self.per
        params.rp = self.rp[sensor_id]
        params.a = self.a
        params.inc = self.inc
        params.ecc = self.ecc
        params.w = self.w   
        params.limb_dark = self.limb_dark
        params.u = self.u[sensor_id] 
        model=batman.TransitModel(params, self._times[sensor_id], exp_time=(self._times[sensor_id][1]-self._times[sensor_id][0]), supersample_factor=self.supersample_factor, max_err=1)
        res = model.light_curve(params)
        res *= np.polynomial.chebyshev.chebval(self._times_norm[sensor_id], self.poly_vals[sensor_id])        
        return res
        
    def _infer_single(self, data):
        print('sanity checks!')
        # Load data
        data.transits[0].load_to_step(5, data, self.loaders)
        
        # Prepare stuff       
        self.poly_order = self.poly_order_step1
        self.per = data.P*24*3600
        self.t0 = 3.5*3600
        self.a = data.sma
        self.inc = data.i
        self.ecc = data.e
        self.poly_vals = [np.zeros(self.poly_order+1), np.zeros(self.poly_order+1)]        
        targets = [None, None]
        self._times = [None, None]
        self._times_norm = [None, None]
        for ii in range(2):
            self.poly_vals[ii][0] = 1
            targets[ii] = cp.mean(data.transits[0].data[ii].data, axis=1)
            targets[ii] /= cp.mean(targets[ii])
            targets[ii] = targets[ii].get()
            self._times[ii] = data.transits[0].data[ii].times.get()
            t=self._times[ii]
            self._times_norm[ii] = 2*(t - t.min())/(t.max() - t.min()) - 1
            
        # Step 1
        for ii in range(2):
            self.t0 = np.max(self._times[ii])/2
            target_cp = cp.array(targets[ii])            
            base_curve = self._light_curve(ii)
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
            self.poly_vals[ii] = cp.linalg.lstsq(design_mat, target_cp)[0].get()            
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
                cost += np.sqrt(np.mean( (self.pred[ii]-targets[ii])**2 ))
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
            #print(cost(x0))
            res = scipy.optimize.minimize(cost,x0)
            #print(cost(res.x))
            self.cost_list.append(cost(res.x))
                    
        
        if self.do_plots:            
            _,ax = plt.subplots(1,3,figsize=(15,5))
            for ii in range(2):
                plt.sca(ax[ii])
                plt.grid(True)
                plt.xlabel('Time [h]')
                plt.ylabel('Normalized intensity')
                plt.plot(self._times[ii]/3600, targets[ii])
                plt.plot(self._times[ii]/3600, step1[ii])
                plt.plot(self._times[ii]/3600, self.pred[ii])
                plt.legend(('Measured', 'After step 1', 'After step 2'))
            plt.sca(ax[2])
            plt.plot(self.order_list[1:], self.cost_list[1:])
            plt.grid(True)
            plt.xlabel('Poly order')
            plt.ylabel('Cost')
            plt.pause(0.001)
            
        self._times = None
        self._times_norm = None
                
        return data
    