'''
This code is released under the CC BY 4.0 license, which allows you to use and alter this code (including commercially). You must, however, ensure to give appropriate credit to the original author (Jeroen Cottaar). For details, see https://creativecommons.org/licenses/by/4.0/
'''

#from pandas.io.clipboard import copy; copy("\x04")
import pandas as pd
import numpy as np
import dill as pickle
from astropy.stats import sigma_clip
import ariel_support as ars
import itertools
import os
import pandas.api.types
import scipy.stats
import scipy as sp
import scipy.ndimage
import ariel_gp as arg
import multiprocess as multiprocessing
import copy
import importlib
import gp

def run_all_tests():
    if len(ars.test_planet_list)==1:
        loader_test()
    regression_test()
    
def regression_test(update_reference=False):

    importlib.reload(ars)
    importlib.reload(gp)
    importlib.reload(arg)
    case_names = ['Baseline', 'All options changed', 'No transit update, no PCA']
    res = pd.DataFrame(columns=['case', 'rms_error', 'score', 'match'])
    loader = ars.DataLoader()
    loader.loader_options = ars.baseline_loader()
    loader.planet_ids_to_load = loader.planet_ids_to_load[:3]
    loader.use_cache = not update_reference
    print(loader.planet_ids_to_load)
    data_base = loader.load()
    for i in range(len(case_names)):
        if i == 0:
            model = arg.baseline_model()
        elif i==1:
            model = arg.baseline_model()           
            model.model.model.model_options.min_transit_scaling_factor = 2*model.model.model.model_options.min_transit_scaling_factor
            model.model.model.model_options.n_pca = 2
            model.model.model.model_options.retrain_pca = not model.model.model.model_options.retrain_pca
            model.model.model.model_options.n_samples_sigma_est = 50
            model.model.model.model_options.hfactor = 3           
            model.model.model.model_options.n_iter=2
            model.model.model.model_options.update_rate = 0.5
            model.model.model.model_options.max_log_update_hyperparameters = 2
        elif i==2:
            model = arg.baseline_model()
            model.model.model.model_options.update_transit_variation_sigma = not model.model.model.model_options.update_transit_variation_sigma        
            model.model.model.model_options.n_pca = 0
            model.model.model.model_options.n_iter=1
        
            
        model.run_in_parallel= True
        data = data_base        
        model.train(data)
        pred,sigma,cov = model.infer(data)
        score_val, rms_error = ars.score_wrapper([d['planet_id'] for d in data], pred, sigma)
        res.loc[i] = [case_names[i], 1e4*rms_error, score_val, None]
    if update_reference:
        ars.pickle_save(ars.file_loc()+'regression.pickle', res)
        print('Result and new reference')
        print(res)
    else:
        ref = ars.pickle_load(ars.file_loc()+'regression.pickle')       
        for i in range(len(case_names)):
            tol = 1e-3
            if ars.running_on_kaggle == 1:
                tol = 1e-3
            res.at[i, 'match'] = (np.abs(res.iloc[i]['rms_error']-ref.iloc[i]['rms_error']))<tol and (np.abs(res.iloc[i]['score']-ref.iloc[i]['score']))<tol
        print('Result')
        print(res)
        print('Reference')
        print(ref) 
        if not all(res['match']):
            raise Exception('Regression failed')
        #print(res['match'])
        #print(all(res(['match'])))
    print('Regression passed')
    return res

def loader_test(update_reference=False):
    importlib.reload(ars)
    loader = ars.DataLoader()
    loader.loader_options = ars.baseline_loader()
    loader.planet_ids_to_load = [loader.planet_ids_to_load[100]]
    loader.use_cache = False
    results = dict()
    results[0] = loader.load()
    loader.loader_options.options_AIRS.correct_jitter = not loader.loader_options.options_AIRS.correct_jitter
    loader.loader_options.options_AIRS.inpainting = not loader.loader_options.options_AIRS.inpainting
    loader.loader_options.options_AIRS.column_binning  = -loader.loader_options.options_AIRS.column_binning
    loader.loader_options.options_FGS.column_binning  = -loader.loader_options.options_FGS.column_binning
    results[1] = loader.load()
    if update_reference:
        ars.pickle_save(ars.file_loc()+'regression_load.pickle', results)
    else:
        results_ref = ars.pickle_load(ars.file_loc()+'regression_load.pickle')
        assert str(results) == str(results_ref)
    
        

