#program to look at power spectra for Parachute
#First created: January 2024
#Author: Paul Burns



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






suite 	= 'dc366'
basedir = '/data/users/pburns/cylc-run/'
datadir = f'{basedir}u-{suite}/share/'
plotDir = '../plots/'

# field_type = 'a'  # 1D profiles
field_type = 'r' # Full 3D fields

z_grid_type = 'L70a'

vars = {
        'm01s00i150': (r'$w$', 'm s$^{-1}$'),
        'm01s00i010': (r'$q$', 'kg kg$^{-1}$'),
       }

dxs = ['200m','500m','1km','10km']

z_slices = [400]

window_type = 'boxcar'
#window_type = 'tukey'
#Since we use a periodic domain it is safe for the FFT to assume
#that our signal is periodic and repeat the signal.

colours		= ['k','r','g','b']
linestyles	= ['solid','dashdot','dashed',(0, (3, 5, 1, 5, 1, 5)),'dotted']




def compute_psd(dat, Nx, dx, window_type):

    dx = int(re.findall(r'\d+', dx)[0])
    if dx == 1: dx = 1000
    if dx == 10: dx = 10000 

    if window_type == 'blackman': weights = blackman(Nx)
    if window_type == 'tukey': weights = tukey(Nx)
    if window_type == 'boxcar': weights = boxcar(Nx)

    coeffs = fft(dat*weights)

    psd = np.zeros((Nx//2))
    psd[0] = 1./Nx * np.abs(coeffs[0])**2
    psd[1:] = 2.0/Nx * np.abs(coeffs[1:Nx//2])**2

    freq = fftfreq(Nx, dx)[:Nx//2]

    return psd, freq



def plot_spectra(ds, dxs, vars, z_slices, window_type):

    fig, axes = plot_env.initialise(rows=1,cols=len(vars), aspect_ratio=2/3.)
    fig.set_tight_layout(True)

    if len(vars) == 1: axes = [axes]

    for ix, stash_code in enumerate(vars):
        name, units = vars[stash_code]
        ax = axes[ix]

        ax.set_xscale('log')
        ax.set_yscale('log')

        #ax.set_ylim(1E-13,1E0)
        ax.set_xlabel(r'$k$ (1/m)')
        #ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$k\times$Power ()')
        #if ix > 0: ax.get_yaxis().set_visible(False)

        for i, dx in enumerate(dxs):

            label = f"{dx}"
            field = ds[i]['STASH_'+stash_code]

            clon = [c for c in field.coords if 'longitude' in c][0]
            clat = [c for c in field.coords if 'latitude' in c][0]
            ceta = [c for c in field.coords if 'eta_theta' in c][0]
            field = field.rename({clon:'x', clat:'y', 'min15T0':'time', ceta:'eta_theta'}) 

            Ny = len(field.coords['y'].values)
            Nx = len(field.coords['x'].values)

            zyx_slice = field.mean(dim='time')

            if ix==0 and i==0:
                H = 40000
                z = zyx_slice.coords['eta_theta'].values*H
                
            for j, probe_z in enumerate(z_slices):

                idxs            = np.where( z <= probe_z )
                idxZ      	= np.max(idxs)
                yx_slice 	= zyx_slice.isel(eta_theta = idxZ)

                psd_y = np.zeros((Ny,Nx//2))
                for yy in range(Ny):
                    x_series = yx_slice.isel(y=yy)
                    psd_yy, freq = compute_psd(x_series.values, Nx, dx, window_type)
                    psd_y[yy,:] = psd_yy
                
                psd_Ymean = np.mean(psd_y,axis=0)

                psd_x = np.zeros((Ny//2,Nx))
                for xx in range(Nx):
                    y_series = yx_slice.isel(x=xx)
                    psd_xx, freq = compute_psd(y_series.values, Ny, dx, window_type)
                    psd_x[:,xx] = psd_xx

                psd_Xmean = np.mean(psd_x,axis=1)


                ax.plot(freq[1:], np.multiply(psd_Xmean[1:],freq[1:]),\
                             color=colours[j], linestyle=linestyles[i], label=label)

                if ix==0 and i==0 and j==0 : ax.plot(freq[1:], freq[1:]**(-5/3.)*1E-8, 'b:')
                if ix==1 and i==0 and j==0 : ax.plot(freq[1:], freq[1:]**(-5/3.)*1E-22, 'b:')

        ax.legend(frameon=False, ncol=1, handlelength=3, fontsize=10)


    plt.savefig(f'{plotDir}psd_{suite}_{field_type}.eps')

    for d in ds: d.close()
    return fig




def main():
    ds = readXARR(datadir, suite, dxs, field_type, z_grid_type)
    fig = plot_spectra(ds, dxs, vars, z_slices, window_type)


if __name__ == '__main__':
    main()



