import kaggle_support as kgs
import ariel_simple
import ariel_gp
from dataclasses import dataclass, field, fields
import scipy
import numpy as np
import cupy as cp
import copy
import time
import matplotlib.pyplot as plt

@dataclass
class PCA(kgs.Model):    
    model: kgs.Model = field(init=True, default=None)    
    model_options_link: ariel_gp.ModelOptions = field(init=True, default=None)
    learn_pca_during_training: bool = field(init=True, default=True)   
    
    n_components: int = field(init=True, default=5)
    ignore_FGS = False
    rescale = True
    rescale_gamma = 0.5
    filter_ratio = 1.
    
    adapt_gp_scaling = True
    FGS_target = 0.0001
    AIRS_target = 0.0001
  
    def _train(self,train_data):
        if self.learn_pca_during_training:
            spectra = np.array([d.spectrum for d in train_data])
            self._learn_pca(spectra)
        self.model.train(train_data[:12])        
        
    
    def _infer(self,data):
        return self.model.infer(data)
    
    def _learn_pca(self, spectra):
        
        model_options = self.model_options_link
        
        # Do the PCA
        X = spectra-np.mean(spectra,1)[:,None]        
        std_vals = np.std(X,1)       
        inds = np.argsort(std_vals)
        X = X[inds[np.round((1-self.filter_ratio)*X.shape[0]).astype(int):],:]
        if self.rescale:
            X = X / np.std(X,1)[:,None]**self.rescale_gamma
        #print('b', np.std(X[:,0])/np.std(X[:,1:]))
        X[:,0] = X[:,0]/np.std(X[:,0])*self.FGS_target
        X[:,1:] = X[:,1:]/np.std(X[:,1:])*self.AIRS_target
        #print('a', np.std(X[:,0])/np.std(X[:,1:]))
       
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        # Principal directions in feature space (rows = PCs)
        components = Vh                              # shape: (n_components, n_features)
        components = components[:self.n_components,:]
        
        if self.ignore_FGS:
            components[:,0] = 0

        # Weights / scores of each sample on each PC
        weights = U * S                              # shape: (n_samples, n_components)
        weights = weights[:,:self.n_components]
        
        pred = weights @ components
        
        self.model_options_link.transit_pca_components = components
        #print(components.shape)
        self.model_options_link.transit_pca_variances = np.var(weights,0)
        #print(np.var(weights,0))
        
        #plt.figure()
        #plt.plot(components.T)

        if self.adapt_gp_scaling:
            # Find proper AIRS scaling
            #print(X.shape)
            ratio = np.var(pred[:,1:])/np.var(X[:,1:])
            #print(1-ratio)
            self.model_options_link.AIRS_transit_scaling *= np.sqrt(1-ratio)

            # Find proper FGS scaling
            ratio = np.var(pred[:,0])/np.var(X[:,0])
            #print(1-ratio)
            self.model_options_link.FGS_transit_scaling *= np.sqrt(1-ratio)
        
        n_samples = X.shape[0]
        den = max(n_samples - 1, 1)                  # avoid /0 if there's only 1 sample
        explained_variance = (S**2) / den            # per-component variance (like sklearn's PCA.explained_variance_)
        explained_variance_ratio = (S**2) / np.sum(S**2)

        # Your plots
        #plt.figure()
        #plt.plot(wavelengths, components[:5, :].T)   # first 5 PCs as spectra vs wavelength

        #plt.figure()
        #plt.plot(1 - np.cumsum(explained_variance_ratio)[:20])  # leftover variance (like your original)
        #plt.semilogy(explained_variance_ratio[:20])
        #plt.pause(0.001)
        #(1 - np.cumsum(explained_variance_ratio)[:20])[0]
        #kgs.rms(X),kgs.rms(X-weights@components)

        
        
        #self.model_options_link.FGS_transit_scaling *= 2
        #self.model_options_link.AIRS_transit_scaling *= 10

