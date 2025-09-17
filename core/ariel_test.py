import kaggle_support as kgs
import importlib
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import ariel_load
import ariel_simple
import ariel_model
import ariel_gp
import dill
import deepdiff
import copy


def run_all_tests(regenerate_reference=False):
    run_loader_test(use_cache=False, regenerate_reference=regenerate_reference)
    run_loader_test(use_cache=True, regenerate_reference=False)
    run_loader_test(use_cache=True, regenerate_reference=False) # twice in case first one wrote cache
    run_model_test(regenerate_reference=regenerate_reference)
    print('All tests passed!')
    
    
    
    
    
    
    
def run_loader_test(use_cache=False, regenerate_reference=False):
    kgs.debugging_mode = 2
    train_data = kgs.load_all_train_data()
    data = train_data[75]
    loaders = ariel_load.default_loaders()
    if not use_cache:
        for ii in range(2):
            loaders[ii].cache_steps = []
    data.load_to_step(5, loaders)
    data = (data.transits[0].data[0].data.get(),(data.transits[1].data[1].data.get()))
    if regenerate_reference:
        kgs.dill_save(kgs.code_dir + '/loader_test.pickle', data)
    diff=deepdiff.DeepDiff(data, kgs.dill_load(kgs.code_dir + '/loader_test.pickle'))
    if len(diff)>0:
        print(diff)
        raise Exception('Mismatch in loader test')

def run_model_test(regenerate_reference=False):
    kgs.debugging_mode = 2
    models = dict()
    models['simple'] = ariel_simple.SimpleModel()    
    models['simple'].run_in_parallel = not regenerate_reference
    models['baseline'] = ariel_model.baseline_model()
    models['baseline'].model.n_components = 1
    models['baseline'].model.model.run_in_parallel = not regenerate_reference
    train_data = kgs.load_all_train_data()
    for name,model in models.items():
        data = copy.deepcopy(train_data[38:41])
        for d in data:
            copy.deepcopy(d).load_to_step(5, ariel_load.default_loaders())
        #model.run_in_parallel = not regenerate_reference
        #try: model.model.run_in_parallel = not regenerate_reference
        #try: model.model.modrun_in_parallel = not regenerate_reference
        model.train(data)
        data = model.infer(data)
        if regenerate_reference:
            kgs.dill_save(kgs.code_dir + '/model_'+name+'_test.pickle', data)
        else:
            for d in data:
                del d.diagnostics['sanity_checks_par']
        diff=deepdiff.DeepDiff(data, kgs.dill_load(kgs.code_dir + '/model_'+name+'_test.pickle'))
        if len(diff)>0:
            print(diff)
            raise Exception('Mismatch in ' + name + ' test')
