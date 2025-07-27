import numpy as np
import cupy as cp
import kaggle_support as kgs
import scipy
import copy
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field, fields
from matplotlib import animation, rc; rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
matplotlib.rcParams['animation.html'] = 'jshtml'


def animate_3d_matrix(animation_arr, fps=20, figsize=(6,6), axis_off=False, title='', cmap='viridis'):

    animation_arr= copy.deepcopy(animation_arr[...])
    
    # Initialise plot
    fig = plt.figure(figsize=figsize)  # if size is too big then gif gets truncated

    min_val = np.nanmin(animation_arr)
    max_val = np.nanmax(animation_arr)
    
    im = plt.imshow(animation_arr[0], cmap=cmap, aspect='auto')
    plt.clim([min_val,max_val])
    plt.colorbar()
    plt.title(title)
    if axis_off:
        plt.axis('off')
    #plt.title(f"{tomo_id}", fontweight="bold")

    
    #print('range: ', min_val,max_val)
    #animation_arr = (animation_arr-min_val)/(max_val-min_val)
    # Load next frame
    def animate_func(i):
        im.set_data(animation_arr[i])
        #plt.clim([0, 1])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)

    display(anim)
    return
