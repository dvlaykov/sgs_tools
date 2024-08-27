# Program to plot flux components for Parachute
# First created: January 2024
# Author: Paul Burns



import plot_env
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math
from decimal import Decimal
import pdb




def log_linsub(start, end):
    """
    Creates a log scale from 10**start to 10**end with linear subdivisions.
    """
    return np.unique([base * 10**val for val in range(start, end) for base in range(1, 11)])



def plot_scalar_flux(dat_in, dxs, t0, times, scalar, plotBaseDir, geostrophic, eff_res_fact, z_lim=None,\
                     plt_localFlux=True, plt_nonlocalFlux=False,\
                     plt_subgrid_total=False, plt_subgrid_total_um=False,\
                     plt_leonard=False, plt_resolved=False, plt_total=False, plt_flux_grad=False,\
                     plt_w=False, plt_updrafts=False, plt_randomPick=False, n=50, plt_cg_filtered=False,\
                     z_profile=True, z_slice=False, xy_pnt=None, z_level=None, count=None,\
                     limitDomain=False, cbar_type=0):

    if z_profile: aspect_ratio = 2/3.
    #if z_slice: aspect_ratio = 2/4.
    if (len(dxs)==4 or len(dxs)==3) and z_slice: aspect_ratio = 2/6.
    #if (len(dxs)==4 or len(dxs)==3) and z_slice: aspect_ratio = 2.5/6
    if len(dxs)==1 and z_slice: aspect_ratio = 1.

    rows = 1
    cols = len(dxs)

    fig, subfig, axes = plot_env.initialise(rows=rows,cols=cols, aspect_ratio=aspect_ratio)
    fig.set_tight_layout(True)

    if len(dxs)==1: axes = [axes]

    if len(times)>1: dat = dat.mean(dim='min15T0')

    for i, dx in enumerate(dxs):
        print('dx: ', dx)

        dat = dat_in[i]

        if z_profile and i>0: axes[i].get_yaxis().set_visible(False)

        dx_ = dat.longitude_t.values[1]-dat.longitude_t.values[0]
        Lx = np.max(dat.longitude_t.values)

        if z_profile:

            if xy_pnt==None and xy_pnt != 'updrafts_mean':
                dat = dat.mean(dim=['longitude_t','latitude_t'])
                dat = dat.mean(dim=['longitude_cu','latitude_cu'])
                dat = dat.mean(dim=['longitude_cv','latitude_cv'])
            if xy_pnt is not None and xy_pnt != 'updrafts_mean':
                if xy_pnt=='central':
                    idx_lon = int(len(dat.longitude_t)/2.)
                    idx_lat = int(len(dat.latitude_cv)/2.)
                if xy_pnt=='pnt1':
                    idx_lon = int(len(dat.longitude_t)/4.)
                    idx_lat = int(len(dat.latitude_cv)/4.)
                if xy_pnt=='random':
                    Nx = len(dat.longitude_t)-1
                    Ny = len(dat.latitude_t)-1
                    idx_lon = int(float(Nx)/rdm_num)
                    idx_lat = int(float(Ny)/rdm_num)
                if xy_pnt=='updrafts':
                    fnm_common = str(n) + '_' + str(int(dx_)) + '_' + geostrophic
                    if plt_cg_filtered and int(dx)!=100: fnm_common = fnm_common + '_filtered'
                    updrafts_rdm_xIdxs = np.load('../output/updrafts/updrafts_rdm_xIdxs_'\
                                                 + fnm_common + '.npy', allow_pickle=False)
                    updrafts_rdm_yIdxs = np.load('../output/updrafts/updrafts_rdm_yIdxs_'\
                                                 + fnm_common + '.npy', allow_pickle=False)
                    idx_lon = int(updrafts_rdm_xIdxs[count])
                    idx_lat = int(updrafts_rdm_yIdxs[count])

                dat = dat.isel(latitude_t=idx_lat, longitude_t=idx_lon)
                dat = dat.isel(latitude_cu=idx_lat, longitude_cu=idx_lon)
                dat = dat.isel(latitude_cv=idx_lat, longitude_cv=idx_lon)

            z_th = dat.thlev_eta_theta.values
            z_rho = dat.rholev_eta_rho.values

            if plt_localFlux:
                var = 'F_Sx_th'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_th,\
                          '-.x', color='k', markerfacecolor='none', markersize=7, label=r'$F_{{\rm S},x}$')

                var = 'F_Sy_th'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_th,\
                          '-.+', color='k', markerfacecolor='none', markersize=15, label=r'$F_{{\rm S},y}$')

                var = 'F_Sz_rho'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                          '-.^', color='k', markerfacecolor='none', markersize=7, label=r'$F_{{\rm S},z}$')

                if xy_pnt == 'updrafts_mean': 
                    lwr_bound = dat[var+sfx] - 2*dat[var + '_std']
                    upr_bound = dat[var+sfx] + 2*dat[var + '_std']
                    axes[i].fill_betweenx(z_th, lwr_bound, upr_bound,\
                                      color="none", hatch="-", edgecolor="k", alpha=0.25, linewidth=0.5)

            if plt_nonlocalFlux:
                var = 'F_nl_rho'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                          '-.^', color='r', markerfacecolor='none', markersize=7, label=r'$F_{\rm nl}$')
                if xy_pnt == 'updrafts_mean': 
                    lwr_bound = dat[var+sfx] - 2*dat[var + '_std']
                    upr_bound = dat[var+sfx] + 2*dat[var + '_std']
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                      color="none", hatch="-", edgecolor="r", alpha=0.25, linewidth=0.5)

            if plt_subgrid_total:
                var = 'sg_total'
                sfx = ''
                if plt_cg_filtered: sfx = sfx + '_' + str(int(dx_))
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                z_rho = dat[var+sfx].rholev_eta_rho.values

                axes[i].plot( dat[var+sfx], z_rho,\
                              '-.o', color='orange', markerfacecolor='none', markersize=7, label=r'$F_{\rm subgrid~total}$')
                if xy_pnt == 'updrafts_mean':
                    sfx2 = '' 
                    if plt_cg_filtered: sfx2 = sfx2 + '_' + str(int(dx_))
                    if xy_pnt == 'updrafts_mean': sfx2 = sfx2 + '_std'
                    lwr_bound = dat[var+sfx] - 2*dat[var + sfx2]
                    upr_bound = dat[var+sfx] + 2*dat[var + sfx2]
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="-", edgecolor="orange", alpha=0.25, linewidth=0.5)

            if plt_subgrid_total_um:
                var = 'F_blend_expl_rho'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                z_rho = dat[var].rholev_eta_rho.values
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-.o', color='yellow', linewidth=0.5, markerfacecolor='none', label=r'$F_{\rm subgrid~total, UM}$')
                if xy_pnt == 'updrafts_mean': 
                    lwr_bound = dat[var+sfx] - 2*dat[var + '_std']
                    upr_bound = dat[var+sfx] + 2*dat[var + '_std']
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="-", edgecolor="yellow", alpha=0.25, linewidth=0.5)

                var = 'F_blend_grad_rho'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-', color='k', linewidth=0.5, markerfacecolor='none', label=r'$F_{\rm grad, UM}$')
                if xy_pnt == 'updrafts_mean': 
                    lwr_bound = dat[var+sfx] - 2*dat[var + '_std']
                    upr_bound = dat[var+sfx] + 2*dat[var + '_std']
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="-", edgecolor="k", alpha=0.25, linewidth=0.5)

                var = 'F_blend_nongrad_rho'
                sfx = ''
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-', color='r', linewidth=0.5, markerfacecolor='none', label=r'$F_{\rm nl, UM}$')
                if xy_pnt == 'updrafts_mean': 
                    lwr_bound = dat[var+sfx] - 2*dat[var + '_std']
                    upr_bound = dat[var+sfx] + 2*dat[var + '_std']
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="-", edgecolor="r", alpha=0.25, linewidth=0.5)

            if plt_leonard:
                var = 'L_3phi_rho'
                sfx = ''
                if plt_cg_filtered: sfx = sfx + '_' + str(int(dx_))
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                z_rho = dat[var+sfx].rholev_eta_rho.values
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-.^', color='b', markerfacecolor='none', markersize=7, label=r'$L_z$')
                if xy_pnt == 'updrafts_mean':
                    sfx2 = ''
                    if plt_cg_filtered: sfx2 = sfx2 + '_' + str(int(dx_))
                    if xy_pnt == 'updrafts_mean': sfx2 = sfx2 + '_std'
                    lwr_bound = dat[var+sfx] - 2*dat[var + sfx2]
                    upr_bound = dat[var+sfx] + 2*dat[var + sfx2]
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="-", edgecolor="b", alpha=0.25, linewidth=0.5)

            if plt_resolved:
                z_th = dat.thlev_eta_theta.values
                z_rho = dat.rholev_eta_rho.values
                #axes[i].plot( dat['F_res_x'], z_th,\
                #             '-.x', color='g', markerfacecolor='none', markersize=7, label=r'$\overline{u}\,\overline{\theta}$')
                #axes[i].plot( dat['F_res_y'], z_th,\
                #             '-.+', color='g', markerfacecolor='none', markersize=7, label=r'$\overline{v}\,\overline{\theta}$')

                var = 'F_res_z'
                sfx = ''
                if plt_cg_filtered: sfx = sfx + '_' + str(int(dx_))
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-.^', color='g', markerfacecolor='none', markersize=7, label=r'$F_{\rm res}$')
                if xy_pnt == 'updrafts_mean':
                    sfx2 = '' 
                    if plt_cg_filtered: sfx2 = sfx2 + '_' + str(int(dx_))
                    if xy_pnt == 'updrafts_mean': sfx2 = sfx2 + '_std'
                    lwr_bound = dat[var+sfx] - 2*dat[var + sfx2]
                    upr_bound = dat[var+sfx] + 2*dat[var + sfx2]
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="///", edgecolor="g", alpha=0.25, linewidth=0.5)

            if plt_total:
                z_th = dat.thlev_eta_theta.values
                z_rho = dat.rholev_eta_rho.values

                var = 'total'
                sfx = ''
                if plt_cg_filtered: sfx = sfx + '_' + str(int(dx_))
                if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
                axes[i].plot( dat[var+sfx], z_rho,\
                              '-.^', color='lightgray', markerfacecolor='none', markersize=7, label=r'$F_{\rm total}$')
                if xy_pnt == 'updrafts_mean':
                    sfx2 = ''
                    if plt_cg_filtered: sfx2 = sfx2 + '_' + str(int(dx_))
                    if xy_pnt == 'updrafts_mean': sfx2 = sfx2 + '_std'
                    lwr_bound = dat[var+sfx] - 2*dat[var + sfx2]
                    upr_bound = dat[var+sfx] + 2*dat[var + sfx2]
                    axes[i].fill_betweenx(z_rho, lwr_bound, upr_bound,\
                                          color="none", hatch="///", edgecolor="lightgray", alpha=0.25, linewidth=0.5)


            #if xy_pnt=='updrafts_mean' and plt_cg_filtered: axes[i].set_xlim(-200,600)
            axes[i].set_xlim(-150,350)
            #axes[i].set_xlim(-500,500)

            xbnds = axes[i].get_xbound()
            var = 'z_bl'
            sfx = ''
            if plt_cg_filtered: sfx = sfx + '_' + str(int(dx_))
            if xy_pnt == 'updrafts_mean': sfx = sfx + '_mean'
            axes[i].plot( [xbnds[0],xbnds[1]], [dat[var+sfx],dat[var+sfx]],\
                           ':', color='gray', linewidth=1 )

            axes[i].set_ylim(0,np.max(z_th))
            axes[i].plot( [0,0], [0,np.max(z_th)], '-', color='gray', linewidth=2 )
            axes[i].legend(frameon=False, ncol=1, handlelength=3, fontsize=10)

            axes[i].set_xlabel(r'Heat Flux (W m$^{-2}$)')
            axes[i].set_ylabel(r'$z$ (m)')

            if xy_pnt != 'updrafts_mean':
                lat_frac = str(round(idx_lat*dx_/Lx,2))
                lon_frac = str(round(idx_lon*dx_/Lx,2))
            time_str = str((t0 + times[0])/60.)
            title = str(int(dx_)) + ' m' + '\n' + time_str + ' min'
            axes[i].set_title(title, fontsize=10)


        #----------------------------------------------------------------------------------------------#
        #----------------------------------------------------------------------------------------------#

        if z_slice:

            x = dat.longitude_t.values/1000
            y = dat.latitude_t.values/1000
            x_limit = 25 #to limit domain of non LES runs to LES domain for comparions

            if limitDomain and plt_cg_filtered==False:
                idxs = np.where( x <= x_limit )
                xIdxE = np.max(idxs)
                x = x[0:xIdxE]
                idxs = np.where( y <= x_limit )
                yIdxE = np.max(idxs)
                y = y[0:yIdxE]
           
                dat = dat.isel(longitude_t = list(range(0,xIdxE)))
                dat = dat.isel(latitude_t = list(range(0,yIdxE)))
 
            if plt_updrafts == False:
                if z_level==None:
                    dat = dat.mean(dim=['rholev_eta_rho'], skipna=True)
                    dat = dat.mean(dim=['thlev_eta_theta'], skipna=True)
                    #dat = dat.mean(dim=['thlev_bl_eta_theta'], skipna=True)
                if isinstance(z_level, numbers.Number):
                    idxs = np.where(dat.rholev_eta_rho.values <= z_level)
                    zIdx = np.max(idxs)
                    dat = dat.isel(rholev_eta_rho=zIdx)
                    dat = dat.isel(thlev_eta_theta=zIdx)
                if z_level == 'z_abl':
                    z_bl_mean = dat['z_bl'].mean(dim=['longitude_t','latitude_t'], skipna=True).values
                    print(z_bl_mean)
                    idxs = np.where(dat.rholev_eta_rho.values <= z_bl_mean)
                    zIdx = np.max(idxs)
                    dat = dat.isel(rholev_eta_rho=zIdx)
                    dat = dat.isel(thlev_eta_theta=zIdx)

                if plt_localFlux: 
                    var = 'F_Sz_rho'
                    max_f = 60
                if plt_nonlocalFlux: 
                    var = 'F_nl_rho'
                    max_f = 100
                if plt_subgrid_total: 
                    var = 'sg_total'
                    max_f = 180
                if plt_leonard:
                    var = 'L_3phi_rho'
                    max_f = 50
                if plt_resolved:
                    var = 'F_res_z'
                    max_f = 0.3E6
                if plt_total: 
                    var = 'total'
                    max_f = 0.1E6
                if plt_w:
                    var = 'w'
                    max_f = 1E0

                if plt_cg_filtered: var = var + '_' + str(int(dx_))
                Nlevs = 20+1  
                #fctr = 10.
                fctr = 1.
                max_f = math.ceil(np.max(dat[var].data)/fctr)
                max_f = max_f*fctr
                min_f = -max_f
                df = (max_f-min_f)/(Nlevs-1)
                levels = np.arange(Nlevs)*df + min_f
                print('levels: ', levels)
                im = axes[i].contourf(x,y, dat[var], levels, cmap='bwr')


            if plt_updrafts:
                var = 'updrafts'
                extent = [np.min(x),np.max(x),np.min(y),np.max(y)]

                axes[i].imshow(dat[var], origin='lower',\
                               cmap='Greys', interpolation='nearest', extent=extent)
 
                if plt_randomPick:
                    fnm_common = str(n) + '_' + str(int(dx_)) + '_' + geostrophic
                    if plt_cg_filtered: fnm_common = fnm_common + '_filtered'

                    updrafts_rdm_xIdxs = np.load('../output/updrafts/updrafts_rdm_xIdxs_' + fnm_common + '.npy', allow_pickle=False)
                    updrafts_rdm_yIdxs = np.load('../output/updrafts/updrafts_rdm_yIdxs_' + fnm_common + '.npy', allow_pickle=False)
                    if limitDomain == False:
                        axes[i].scatter(updrafts_rdm_xIdxs*dx_/1000.+np.min(x),\
                                        updrafts_rdm_yIdxs*dx_/1000.+np.min(y), marker='+', c='r', s=40)
                    if limitDomain:
                        for nn in range(0,n):
                            x_nn = updrafts_rdm_xIdxs[nn]*dx_/1000 + np.min(x)
                            y_nn = updrafts_rdm_yIdxs[nn]*dx_/1000 + np.min(y)
                            if x_nn <= np.max(x) and y_nn <= np.max(y):
                                axes[i].scatter(x_nn, y_nn, marker='+', c='r', s=40)

            axes[i].set_title(str(dx_/1000) + ' km')
            axes[i].set_xlabel(r'$x$ (km)')
            if i==0: axes[i].set_ylabel(r'$y$ (km)')

            if limitDomain or plt_cg_filtered:
                axes[i].set_xlim(0,25)
                axes[i].set_ylim(0,25)

            if cbar_type==0 and plt_updrafts==False:
                max_abs_val = float(np.max(np.abs(dat[var].data)))
                if limitDomain or plt_cg_filtered:
                    x_loc = 2.5
                    y_loc = 20
                else:
                    x_loc = Lx/1000*1/8.
                    y_loc = Lx/1000*7/8.
                axes[i].text(x_loc, y_loc, str('%.2E'%Decimal(max_abs_val)), fontsize=8)

            if cbar_type==1 and plt_updrafts==False: 
                cbar = fig.colorbar(im, location='bottom')
                cbar.formatter.set_scientific(True)
                cbar.formatter.set_powerlimits((-1, 1))
                cbar.ax.tick_params(rotation=0)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.locator_params(nbins=4)

    # Add a common colourbar
    if z_slice and cbar_type==2 and plt_updrafts==False: 
        cbar = subfig.colorbar(im, ax=axes, location='right', label=r'W m$^{-2}$')
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((-1, 1))


    # Save figures
    baseDir = plotBaseDir + geostrophic + '/'

    if z_slice:
        if plt_localFlux: fnm = baseDir + 'local_flux'
        if plt_nonlocalFlux: fnm = baseDir + 'nonlocal_flux'
        if plt_subgrid_total: fnm = baseDir + 'sg_total'
        if plt_leonard: fnm = baseDir + 'leonard_flux'
        if plt_resolved: fnm = baseDir + 'resolved'
        if plt_total: fnm = baseDir + 'total_fluxes'
        if plt_updrafts: fnm = baseDir + '/updrafts/updrafts'
        if plt_w: fnm = baseDir + 'w'
        if z_level is None: fnm = fnm + '_z_mean'
        else: fnm = fnm + '_' + str(z_level)
        fnm = fnm + '_' + str(int(times[0] + t0)) + '_' + geostrophic
        if plt_cg_filtered: fnm = fnm + '_filtered'
        fnm = fnm + '_' + str(eff_res_fact)
        if limitDomain and plt_cg_filtered==False: fnm = fnm + '_subdomain'
        fnm = fnm + '.eps'
        #fnm = baseDir + 'tmp.eps'
        plt.savefig(fnm)

    if z_profile:
        if xy_pnt != 'updrafts': 
            fnm = baseDir + 'scalar_fluxes_z_profile'
        if xy_pnt == 'updrafts':
            fnm = baseDir + 'updrafts/scalar_fluxes_updrafts_z_profile_' + str(format(count,'02'))
        if xy_pnt == 'updrafts_mean': 
            fnm = baseDir + 'updrafts/scalar_fluxes_updrafts_mean_profile'
        fnm = fnm + '_' + str(times[0]+t0)
        if plt_cg_filtered: fnm = fnm + '_filtered'
        fnm = fnm + '_' + str(eff_res_fact)
        fnm = fnm + '.eps'
        plt.savefig(fnm)


