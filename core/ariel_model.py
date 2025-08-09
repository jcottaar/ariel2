import kaggle_support as kgs
import ariel_simple
from dataclasses import dataclass, field, fields
import scipy
import numpy as np
import copy
import time

@dataclass
class Fudger(kgs.Model):
    bias_a: list = field(init=True, default_factory=lambda:[1.,1.]) # FGS, AIRS
    bias_b: list = field(init=True, default_factory=lambda:[0.,0.])
    sigma_fudge : list = field(init=True, default_factory=lambda:[1.,1.])
    
    model: kgs.Model = field(init=True, default=None)
    
    _cached_planet_id = None
    _cached_result = None
    
    def _to_x(self):
        return np.reshape([self.bias_a + self.bias_b + self.sigma_fudge], (-1,))
    def _from_x(self,x):
        self.bias_a[0] = x[0]
        self.bias_a[1] = x[1]
        self.bias_b[0] = x[2]
        self.bias_b[1] = x[3]
        self.sigma_fudge[0] = x[4]
        self.sigma_fudge[1] = x[5]
    
    def _train(self,train_data):
        self.model.train(train_data)
        self.state=1
        ii=0
        self.infer(train_data) # dummy
        t=time.time()
        def cost(x):
            self._from_x(x)
            inferred_data = self.infer(train_data)
            nonlocal  ii
            ii+=1
            #print(ii)
            return -kgs.score_metric(inferred_data, train_data, print_results=False)[0]
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
            
        for d in data:
            d.spectrum[0] *= self.bias_a[0]
            d.spectrum[0] += self.bias_b[0]
            d.spectrum[1:] *= self.bias_a[1]
            d.spectrum[1:] += self.bias_b[1]
            d.spectrum_cov[0,:] *= self.sigma_fudge[0]
            d.spectrum_cov[:,0] *= self.sigma_fudge[0]
            d.spectrum_cov[1:,:] *= self.sigma_fudge[1]
            d.spectrum_cov[:,1:] *= self.sigma_fudge[1]
            
        return data

def baseline_model():
    model = Fudger(model=ariel_simple.SimpleModel())
    model.model.run_in_parallel = True
    return model