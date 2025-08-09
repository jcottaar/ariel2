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
class TransitParams(kgs.BaseClass):
    # Holds transit parameters and computes light curves
    Rs:  float = field(init=True, default=None) # Stellar radius in solar radii (R☉).
    Ms:  float = field(init=True, default=None) # Stellar mass in solar masses (M☉).
    Ts:  float = field(init=True, default=None) # Stellar effective temperature in Kelvin.
    Mp:  float = field(init=True, default=None) # Planetary mass in Earth masses (M⊕).
    e:   float = field(init=True, default=None) # Orbital eccentricity (dimensionless).
    w:   float = field(init=True, default=90)    # ???
    Rp:  float = field(init=True, default=0)    # in units of Rs
    P:   float = field(init=True, default=None) # Orbital period in hours.
    t0:  float = field(init=True, default=None) # Transit midpoint in hours.
    sma: float = field(init=True, default=None) # Semi-major axis in stellar radii (Rs), showing the orbital distance relative to the stellar radii.
    i:   float = field(init=True, default=None) # Orbital inclination in degrees.
    limb_dark: str = field(init=True, default = 'quadratic')
    u: np.ndarray = field(init=True, default=None) # limb darkening
    
    # Which parameters are free?
    expose_e_and_w: bool =  field(init=True, default=False)
    
    # Modeling configuration
    supersample_factor = 1
    max_err = 1.
    
    def to_x(self):
        assert not self.expose_e_and_w # todo
        x = [self.t0, self.P, self.sma, self.i,self.Rp]+list(self.u)
        return x
    
    def from_x(self, x):
        x=list(x)
        self.t0 = x[0]
        self.P = x[1]
        self.sma = x[2]
        self.i = x[3]
        self.Rp = x[4]
        self.u = np.array(x[5:])

        if kgs.debugging_mode>=2:
            assert np.all( np.abs(np.array(self.to_x())-np.array(x))<=1e-10 )
            
    
    def light_curve(self,times):
        #print(self,times)
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
        model=batman.TransitModel(params, times, exp_time=(times[1]-times[0]), supersample_factor=self.supersample_factor, max_err=self.max_err)
        return model.light_curve(params)       