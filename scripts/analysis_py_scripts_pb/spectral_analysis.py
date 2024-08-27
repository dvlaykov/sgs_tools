# Program to look at power spectra for Parachute
# First created: May 2024
# Author: Paul Burns



import numpy as np
import matplotlib.pyplot as plt
import plot_env
import pdb #to pause execution use pdb.set_trace()
from readData import readXARR
import sys
import gc
import datetime
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman, tukey, boxcar
import re




def compute_psd(ts, Nx, dx, window_type, normalise_energy):

    if window_type == 'blackman': weights = blackman(Nx)
    if window_type == 'tukey': weights = tukey(Nx)
    if window_type == 'boxcar': weights = boxcar(Nx)

    coeffs = fft(ts*weights)

    # A signal with even number of points has Nyquist frequency in centre
    # A signal with odd number of points does not have a central Nyquist frequency
    if Nx % 2 == 0: Nf = Nx//2 + 1
    else: Nf = Nx//2

    psd = np.zeros((Nf))
    psd[0] = 1./Nx * np.abs(coeffs[0])**2
    psd[1:] = 2.0/Nx * np.abs(coeffs[1:Nf])**2

    freq = np.abs(fftfreq(Nx, d=dx)[:Nf])

    if normalise_energy:
        # Normalise by total energy in signal (Parseval's theorem)
        E_tot = np.sum(psd)
        #print('total energy: ', E_tot)
        psd = psd/E_tot
        #print('normalised total energy: ', np.sum(psd))

    return psd, freq




def mean_spectra(ds, var, dxs, z_level, t0, times,\
                 window_type, psd_axis, normalise_energy, mean_psd,\
                 cg_xy_filter, eff_res_fact, plt_spectra, geostrophic, k5_3rds_line=True):

    z = ds[0].thlev_eta_theta.values

    for i, dx_str in enumerate(dxs):

        dx = int(ds[i].longitude_t.values[1] - ds[i].longitude_t.values[0])
        Ny = len(ds[i].latitude_t.values)
        Nx = len(ds[i].longitude_t.values)

        # Update variable name for filtered results
        if cg_xy_filter: var_ = var + '_' + str(dx)
        else: var_ = var

        if z_level is not None:
            idxs     = np.where( z <= z_level )
            idxZ     = np.max(idxs)
            ds[i] = ds[i].isel(thlev_eta_theta = idxZ)
            ds[i] = ds[i].isel(rholev_eta_rho = idxZ)
            
        if z_level==None:
            ds[i] = ds[i].mean(dim=['rholev_eta_rho'], skipna=True)
            ds[i] = ds[i].mean(dim=['thlev_eta_theta'], skipna=True)

        yx_slice = ds[i][var_]

        # Initialise containers for results
        # Xarray struggles to combine datasets with missing data or subsets of coordinates,
        # so we add data to half the array and discard the other half 
        psd_lon = yx_slice.copy()
        psd_lat = yx_slice.copy()
        freq = yx_slice.copy().isel(longitude_t=0)
        psd_lon.name = 'psd_lon'
        psd_lat.name = 'psd_lat'
        freq.name = 'freq'

        # A signal with even number of points has Nyquist frequency in centre
        # A signal with odd number of points does not have a central Nyquist frequency
        # Even numbers have no remainder when divided by 2
        if Nx % 2 == 0: Nf = Nx//2 + 1
        else: Nf = Nx//2

        for yy in range(Ny):
            x_series = yx_slice.isel(latitude_t=yy)
            psd_yy, freq_ = compute_psd(x_series.values, Nx, dx, window_type, normalise_energy)
            psd_lon.data[yy,0:Nf] = psd_yy
               
        for xx in range(Nx):
            y_series = yx_slice.isel(longitude_t=xx)
            psd_xx, freq_ = compute_psd(y_series.values, Ny, dx, window_type, normalise_energy)
            psd_lat.data[0:Nf,xx] = psd_xx
            if xx==0: freq.data[0:Nf] = freq_

        # Add new power spectra data to dataset
        ds[i] = ds[i].assign(psd_lon=psd_lon)
        ds[i] = ds[i].assign(psd_lat=psd_lat)
        ds[i] = ds[i].assign(freq=freq)



    # simply embed the plotting code here for now
    if plt_spectra:

        baseDir = '../plots/' + geostrophic + '/spectral_analysis/' 

        fig, subfig, ax = plot_env.initialise(rows=1,cols=1, aspect_ratio=4/3., width_factor=0.5)
        fig.set_tight_layout(True)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim(1E-5,1E-2)
        ax.set_ylim(1E-5,1E0)
        if psd_axis=='lon': ax.set_xlabel(r'$k$ (1/m)')
        if psd_axis=='lat': ax.set_xlabel(r'$l$ (1/m)')
        #ax.set_ylabel(r'$k\times$Power ()')
        ax.set_ylabel(r'PSD')

        #colours = []
        #for c in range(len(dxs)): colours.append('k')
        colours = ['k','k','gray','gray']
        linestyles = ['solid','dashdot','dashed',(0, (3, 5, 1, 5, 1, 5)),'dotted']
        #linestyles = ['','','','','']
        #symbols = ['.','o','x','D','s']    
        symbols = ['','','','','']
        
        for i, dx_str in enumerate(dxs):

            dx = ds[i].longitude_t.values[1] - ds[i].longitude_t.values[0]

            Nx = len(ds[i].longitude_t.values)
            if Nx % 2 == 0: Nf = Nx//2 + 1
            else: Nf = Nx//2

            label = str(int(dx))

            if mean_psd:
                if psd_axis == 'lon': dat = ds[i]['psd_lon'].mean(dim='latitude_t').data[0:Nf]
                if psd_axis == 'lat': dat = ds[i]['psd_lat'].mean(dim='longitude_t').data[0:Nf]
            else:
                dat = ds[i]['psd_'+psd_axis].data[0:Nf,int(Nx/2.)]
           
            #if normalise_energy:
            #    # Normalise by total energy in signal (Parseval's theorem)
            #    E_tot = np.sum(dat)
            #    dat = dat/E_tot

            print(dx, np.trapz(dat,ds[i]['freq'].data[0:Nf]))

            #ax.plot(ds[i]['freq'].data[1:Nf], np.multiply(dat,ds[i]['freq'].data[1:Nf]),\
            #             color=colours[i], linestyle=linestyles[i], marker=symbols[i], label=label)
            ax.plot(ds[i]['freq'].data[0:Nf], dat,\
                         color=colours[i], linestyle=linestyles[i], marker=symbols[i], label=label)

            if k5_3rds_line:
                if i==0 and (var=='w' or var=='total'): 
                    ax.plot(ds[i]['freq'].data[10:Nf-10], ds[i]['freq'].data[10:Nf-10]**(-5/3.)*1E-7, 'b', linewidth=0.5)
                if i==0 and var=='phi': 
                    ax.plot(ds[i]['freq'].data[10:Nf-10], ds[i]['freq'].data[10:Nf-10]**(-5/3.)*1E-22, 'b', linewidth=0.5)

        ax.legend(frameon=False, ncol=1, handlelength=3, fontsize=10)

        fnm = baseDir + 'psd_' + var + '_' + str(int(times[0]+t0)) + '_' + geostrophic
        if cg_xy_filter: fnm = fnm + '_filtered' 
        fnm = fnm + '_' + str(eff_res_fact)
        if normalise_energy: fnm = fnm + '_normalised' 
        fnm = fnm + '_' + psd_axis + '.eps'
        plt.savefig(fnm)


