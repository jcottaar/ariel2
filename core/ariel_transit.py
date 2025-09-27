'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module defines transit parameters; it's mostly a wrapper for the batman package.
Using pylightcurve-torch may be an improvement.
'''

import numpy as np
import copy
from dataclasses import dataclass, field
import kaggle_support as kgs
import batman
import math


@dataclass
class TransitParams(kgs.BaseClass):
    # Holds transit parameters and computes light curves
    Rs:  float = field(init=True, default=None) # Stellar radius in solar radii (R☉).
    Ms:  float = field(init=True, default=None) # Stellar mass in solar masses (M☉).
    Ts:  float = field(init=True, default=None) # Stellar effective temperature in Kelvin.
    Mp:  float = field(init=True, default=None) # Planetary mass in Earth masses (M⊕).
    e:   float = field(init=True, default=None) # Orbital eccentricity (dimensionless).
    w:   float = field(init=True, default=90.)    # ???
    Rp:  float = field(init=True, default=0.)    # in units of Rs
    P:   float = field(init=True, default=None) # Orbital period in hours.
    t0:  float = field(init=True, default=None) # Transit midpoint in hours.
    sma: float = field(init=True, default=None) # Semi-major axis in stellar radii (Rs), showing the orbital distance relative to the stellar radii.
    i:   float = field(init=True, default=None) # Orbital inclination in degrees.
    limb_dark: str = field(init=True, default = 'quadratic') # Limb darkening form
    u: np.ndarray = field(init=True, default=None) # Limb darkening parameters
    
    # Modeling configuration
    supersample_factor = 1 # passed to batman
    max_err = 1. # passed to batman
    derivative_step_size = 1e-5 # For finite differences
    

    def _K_hours(self):
        # K such that a_over_Rs = K * P^(2/3); P in hours, Ms in M_sun, Rs in R_sun
        return (((6.67430e-11 * (self.Ms * 1.98847e30) * (3600.0**2)) /
                 (4 * math.pi**2)) ** (1/3)) / (self.Rs * 6.957e8)

    def to_x(self):
        # Conver transit parameters to vector form
        K = self._K_hours()
        A = math.log(self.sma)      # sma is a/Rs (dimensionless)
        B = math.log(self.P)        # P in hours
        alpha = A - B                          # log((a/Rs)/P) -> controls duration/shape
        beta  = A - (2.0/3.0)*B - math.log(K)  # Kepler residual (singular dir)
        x = [self.t0, alpha, beta, self.i, self.Rp] + list(self.u)
        return x

    def from_x(self, x):
        # Inverse of above
        x = list(x)
        self.t0 = x[0]
        alpha   = x[1]
        beta    = x[2]
        self.i  = x[3]
        self.Rp = x[4]
        self.u  = np.array(x[5:])

        # Reconstruct P and sma from (alpha, beta)
        K = self._K_hours()

        # Let A = ln(sma), B = ln(P). Then:
        # alpha = A - B
        # beta  = A - (2/3)B - ln K
        # => B = 3 * (ln K - (alpha - beta)),  A = alpha + B
        B = 3.0 * (math.log(K) - (alpha - beta))
        A = alpha + B

        self.P   = math.exp(B)    # hours
        self.sma = math.exp(A)    # a/Rs

        if kgs.debugging_mode >= 2:
            assert np.all(np.abs(np.array(self.to_x()) - np.array(x)) <= 1e-10)
            
    def sanity_check(self,prefix):
        x = self.to_x()
        kgs.sanity_check(lambda x:x, x[0], prefix+'t0', 5, [2.5,5])
        kgs.sanity_check(lambda x:x, x[1], prefix+'alpha', 6, [-4,-1])
        kgs.sanity_check(lambda x:x, x[2], prefix+'beta', 0, [-1,1])
        kgs.sanity_check(lambda x:x, x[3], prefix+'inc', 7, [84,96])
        kgs.sanity_check(lambda x:x, x[5], prefix+'u0', 8, [0,0.8])
        kgs.sanity_check(lambda x:x, x[6], prefix+'u1', 8, [0,0.4])
            
    
    def light_curve(self,times):
        # Compute the light curve associated with the given parameters
        params = batman.TransitParams()
        params.t0 = self.t0
        params.per = self.P
        params.rp = self.Rp
        params.a = self.sma
        params.inc = self.i
        params.ecc = self.e
        params.w = self.w   
        params.limb_dark = self.limb_dark
        params.u = self.u
        if self.P<=0 or self.sma<=0:
            raise kgs.ArielException(6, 'Bad transit parameters')
        if self.Rp<0:
            params.rp *= -1
        model=batman.TransitModel(params, times, exp_time=(times[1]-times[0]), supersample_factor=self.supersample_factor, max_err=self.max_err)
        res = model.light_curve(params)  
        if self.Rp<0:
            res = 2-res
        assert not np.any(np.isnan(res))
        return res
    
    def light_curve_derivatives(self,times,which):
        # Compute derivatives of the light curve to the parameters, using finite differences
        base_curve = self.light_curve(times)
        res = []
        mod = copy.deepcopy(self)
        x0 = mod.to_x()
        for w in which:            
            x = copy.deepcopy(x0)
            x[w] += self.derivative_step_size
            mod.from_x(x)
            mod_curve = mod.light_curve(times)
            res.append( (mod_curve-base_curve)/self.derivative_step_size )
        return res
            
            