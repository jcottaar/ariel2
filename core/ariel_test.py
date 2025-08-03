import kaggle_support as kgs
import importlib
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import ariel_load
import ariel_simple
import dill
import deepdiff
import copy


def run_all_tests(regenerate_reference=False):
    run_loader_test(use_cache=False, regenerate_reference=regenerate_reference)
    run_loader_test(use_cache=True, regenerate_reference=False)
    run_loader_test(use_cache=True, regenerate_reference=False) # twice in case first one wrote cache
    run_model_test(regenerate_reference=regenerate_reference)
    
    
    
    
    
    
    
def run_loader_test(use_cache=False, regenerate_reference=False):
    train_data = kgs.load_all_train_data()
    data = train_data[75]
    loaders = ariel_load.default_loaders()
    if not use_cache:
        for ii in range(2):
            loaders[ii].cache_steps = []
    data.load_to_step(5, loaders)
    data = (data.transits[0].data[0].data.get(),(data.transits[1].data[0].data.get()))
    if regenerate_reference:
        kgs.dill_save(kgs.code_dir + '/loader_test.pickle', data)
    diff=deepdiff.DeepDiff(data, kgs.dill_load(kgs.code_dir + '/loader_test.pickle'))
    if len(diff)>0:
        print(diff)
        raise Exception('Mismatch in loader test')

def run_model_test(regenerate_reference=False):
    models = dict()
    models['simple'] = ariel_simple.SimpleModel()   
    train_data = kgs.load_all_train_data()
    for name,model in models.items():
        data = copy.deepcopy(train_data[100:102])
        model.run_in_parallel = not regenerate_reference
        model.train(data)
        data = model.infer(data)
        if regenerate_reference:
            kgs.dill_save(kgs.code_dir + '/loader_'+name+'.pickle', data)
        diff=deepdiff.DeepDiff(data, kgs.dill_load(kgs.code_dir + '/loader_'+name+'.pickle'))
        if len(diff)>0:
            print(diff)
            raise Exception('Mismatch in ' + name + ' test')
