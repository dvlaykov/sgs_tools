# Code to initialise python plots
# Author: Paul Burns
# Date of first creation: 15/05/2023

import numpy as np
import matplotlib.pyplot as plt



def initialise(rows=1, cols=1, aspect_ratio=3/4., width_factor=1.):

    #set plot text size
    plt.rcParams.update({'font.size': 12})
    #plt.rcParams.update({'font.size': 24})

    plt.rc('hatch', color='k', linewidth=0.5)

    #set plot size
    a4width         = 8.3
    a4height        = 11.7
    margin          = 0.787402
    width           = (a4width - 2*margin)*width_factor
    height          = width*aspect_ratio

    fig = plt.figure(figsize=(width,height), constrained_layout=True)
    subfig  = fig.subfigures(nrows=1, ncols=1)

    axs = subfig.subplots(rows,cols)


    return fig, subfig, axs



def contour_levels(Nlevels=10, fmin=0, fmax=1):

    f_range = fmax-fmin
    df = f_range/Nlevels
    levels = np.arange(Nlevels)*df + fmin

    return levels



if __name__ == '__main__':
    main()

