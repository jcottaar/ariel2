import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc; rc('animation', html='jshtml')
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

    def animate_func(i):
        im.set_data(animation_arr[i])
        #plt.clim([0, 1])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)

    display(anim)
    return
