# Program to compute scalar (heat or moisture) flux components for Parachute
# First created: January 2024
# Author: Paul Burns

# Code tree: 
#                      |
#                 call main()
#                      |
#                      |-- begin time loop
#                              |
#                              |-- call readXARR() to read in netCDF data for some time
#                              |
#                              |--begin loop over resolutions
#                                    |
#                                    |-- call compute_scalar_flux() to compute fluxes for chosen resolution and time
#                                        and passes back results to main()
#
#                                    |-- choice to call compute_up_down_drafts() to analyse updrafts and downdrafts
#                                        patterns that can then be used to choose location of flux z profiles
#                      |
#                      |-- call plot_scalar_flux() that loops through chosen resolutions and time points
#                                  with options to time and/or space average fields before plotting.
#

# Load code libraries
import numpy as np
import pdb # to pause execution use pdb.set_trace()
import sys
from readData import readXARR_t
from compute_up_down_drafts import compute_up_down_drafts
from plot_results import plot_scalar_flux
from compute_scalar_flux import compute_scalar_flux
from define_vars import define_vars
from flx_component_stats import compute_flx_component_stats
from filters import cg_xy_box_filter
from filters import xy_box_filter
from spectral_analysis import compute_psd, mean_spectra


# Program control (variables with global scope)

# Simulation/data choices
#suite  = 'dc366'
suite   = 'df004'
basedir = '/data/users/pburns/cylc-run/'
datadir = f'{basedir}u-{suite}/share/'

timeUnit = 'timesteps'
#timeUnit = 'minutes'

#geostrophic = 'u_wind'
#geostrophic = 'v_wind'
geostrophic = 'no_wind'

datadir = f'{basedir}u-{suite}/share/{timeUnit}/'
#datadir = f'{basedir}u-{suite}/share/{timeUnit}/{geostrophic}/'
plotBaseDir = '../plots/'

#field_type = 'a'  # 1D profiles
field_type = 'r' # Full 3D fields

z_grid_type = 'L70a'
H = 40000
#t0 = 0.
t0 = 2*60*60
dxs_all = [100,200,500,1000,10000]

if timeUnit == 'timesteps':
    dt_all = [3.,6.,15.,30.,300.]
if timeUnit == 'minutes':
    dt_all = np.repeat(60,len(dxs_all))
    #dt_all = np.repeat(5,len(dxs_all))

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
# Analysis options

times = [0]
#times = [10*60]
#dxs = ['100m','200m','500m','1km']
dxs = ['100m','200m','500m']
#dxs = ['100m']
z_lim = 2000

scalar = 'theta'
fluxes = True
subgrid_total = True
subgrid_total_um = True
leonard = True
resolved = True
total_flx = True
flux_grad = False
computeLocalDiffCoef = False
checkMixingL = True

cg_xy_filter = False
eff_res = False
eff_res_fact = 1
updrafts_downdrafts = False
randomPick = True
Npicks = 50
w2f_randomPick = False
flx_component_stats = False

compute_spectra = False
#psd_var = 'w'
psd_var = 'total'
#psd_axis = 'lon'
psd_axis = 'lat'
normalise_energy = True
mean_psd = True
window_type = 'boxcar'
#window_type = 'tukey'
# Domain is periodic for CBL case so it is safe for the FFT to assume
# that our signal is periodic and repeat the signal.
k5_3rds_line = False

z_level = None
#z_level = 500
#z_level = 'z_abl'

#xy_pnt = None
xy_pnt = 'central'
#xy_pnt = 'pnt1'
#xy_pnt = 'random'
#xy_pnt = 'updrafts'
#xy_pnt = 'updrafts_mean'

if xy_pnt == 'random':
    rdm_num = random.uniform(1,20)
    rdm_num = random.uniform(1,20)

if len(sys.argv)>1 and xy_pnt == 'updrafts':
    count = int(sys.argv[1])
else: count=None


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
# Plot options

plt_flx = True
z_profile = True
z_slice = False
limitDomain = True
plt_localFlux = True
plt_nonlocalFlux = True
plt_subgrid_total = True
plt_subgrid_total_um = True
plt_leonard = False
plt_resolved = False
plt_total = False
plt_flux_grad = False
plt_w = False

plotZgrid = False
plt_updrafts = False
plt_randomPick = True
plt_cg_filtered = False
plt_spectra = False

colours         = ['k','r','g','b']
linestyles      = ['solid','dashdot','dashed',(0, (3, 5, 1, 5, 1, 5)),'dotted']

cbar_type = 0
# 0=print max abs value on plot
# 1=add colourbar for every plot
# 2=add common colourbar





def main():
 
    for tt in range(len(times)):
        print('time loop index: ', tt)

        ds = readXARR_t(datadir, suite, field_type, z_grid_type, dxs, dxs_all, times[tt], dt_all, timeUnit)

        dat_t = ds.copy()

        for i, dx in enumerate(dxs):
            print('dx: ', dx)

            tmp = compute_scalar_flux(ds[i], scalar, H, z_lim=z_lim,\
                                      fluxes=fluxes, subgrid_total=subgrid_total, subgrid_total_um=subgrid_total_um,\
                                      leonard=leonard, resolved=resolved, total_flx=total_flx, flux_grad=flux_grad,\
                                      plotZgrid=plotZgrid, computeLocalDiffCoef=computeLocalDiffCoef, cg_xy_filter=cg_xy_filter,\
                                      compute_spectra=compute_spectra, updrafts_downdrafts=updrafts_downdrafts)

            dat_t[i] = tmp

        if len(times)>1:
            if tt!=0:
                dat = dat_t.copy()
                dat.expand_dims(dim="min15T0")
                dat_t.expand_dims(dim="min15T0")
                dat = xr.concat([dat,dat_t], 'min15T0')

    if len(times) == 1: dat = dat_t

    if cg_xy_filter: dat = cg_xy_box_filter(dat, eff_res_fact)
    if eff_res and eff_res_fact != 1: dat = xy_box_filter(dat, eff_res_fact)

    if updrafts_downdrafts:
        dat = compute_up_down_drafts(dat, dxs, geostrophic, cg_xy_filter,\
                                     z_level=z_level, randomPick=randomPick, n=Npicks, w2f_randomPick=w2f_randomPick)

    if flx_component_stats: dat = compute_flx_component_stats(dat, dxs, geostrophic, cg_xy_filter, Npicks)

    if plt_flx: plot_scalar_flux(dat, dxs, t0, times, scalar, plotBaseDir, geostrophic, eff_res_fact, z_lim=z_lim,\
                     plt_localFlux=plt_localFlux, plt_nonlocalFlux=plt_nonlocalFlux,\
                     plt_subgrid_total=plt_subgrid_total, plt_subgrid_total_um=plt_subgrid_total_um,\
                     plt_leonard=plt_leonard, plt_resolved=plt_resolved,\
                     plt_total=plt_total, plt_flux_grad=plt_flux_grad,\
                     plt_w=plt_w, plt_updrafts=plt_updrafts, plt_randomPick=plt_randomPick, n=Npicks,\
                     plt_cg_filtered=plt_cg_filtered,\
                     z_profile=z_profile, z_slice=z_slice, xy_pnt=xy_pnt, count=count,\
                     limitDomain=limitDomain, cbar_type=cbar_type)

    if compute_spectra: mean_spectra(dat, psd_var, dxs, z_level, t0, times, window_type,\
                                     psd_axis, normalise_energy, mean_psd,\
                                     cg_xy_filter, eff_res_fact, plt_spectra, geostrophic, k5_3rds_line=k5_3rds_line) 


if __name__ == '__main__':
    main()

