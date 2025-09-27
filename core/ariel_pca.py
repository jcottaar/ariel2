'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/

This module implements a PCA analysis of the given transit depths in the training data.
This is used to enhance the transit depth prior in the Bayesian model.
'''
import kaggle_support as kgs
import ariel_gp
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PCA(kgs.Model):    
    model: kgs.Model = field(init=True, default=None)  # The underlying model  
    model_options_link: ariel_gp.ModelOptions = field(init=True, default=None) # Reference to model_options in ariel_gp.PredictionModel

    n_components: int = field(init=True, default=5) # Number of PCA components
    rescale = True # Whether to normalize the transit depth spectra, so the PCA is not dominated by a few big ones
    rescale_gamma = 0.5 # Scaling factor for normalization (see code)
    
    adapt_gp_scaling = True # Whether to adapt the scaling of the non-PCA part of the prior (since the PCA absorbs part of what it should describe)
    FGS_target = 0.0001 # Messy, undocumented
    AIRS_target = 0.0001 # Messy, undocumented
  
    def _train(self,train_data):
        # Do the PCA
        spectra = np.array([d.spectrum for d in train_data])
        self._learn_pca(spectra)
        
        # Train underlying model
        self.model.train(train_data)        
        
    
    def _infer(self,data):
        # Use underlying model to infer
        return self.model.infer(data)
    
    def _learn_pca(self, spectra):
        
        model_options = self.model_options_link
        
        # Collect and normalize data
        X = spectra-np.mean(spectra,1)[:,None]    
        if self.rescale:
            X = X / np.std(X,1)[:,None]**self.rescale_gamma
        X[:,0] = X[:,0]/np.std(X[:,0])*self.FGS_target
        X[:,1:] = X[:,1:]/np.std(X[:,1:])*self.AIRS_target
       
        # Do the PCA. Note that we do not center the data.
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        # Principal directions in feature space (rows = PCs)
        components = Vh                              # shape: (n_components, n_features)
        components = components[:self.n_components,:]
        # Weights / scores of each sample on each PC
        weights = U * S                              # shape: (n_samples, n_components)
        weights = weights[:,:self.n_components]        
        pred = weights @ components # The content described by the PCA
        
        # Adapt prior
        self.model_options_link.transit_pca_components = components
        self.model_options_link.transit_pca_variances = np.var(weights,0)
        
        # Adapt the non-PCA part of the prior
        if self.adapt_gp_scaling:
            # Find proper AIRS scaling
            ratio = np.var(pred[:,1:])/np.var(X[:,1:])
            self.model_options_link.AIRS_transit_scaling *= np.sqrt(1-ratio)

            # Find proper FGS scaling
            ratio = np.var(pred[:,0])/np.var(X[:,0])
            self.model_options_link.FGS_transit_scaling *= np.sqrt(1-ratio)