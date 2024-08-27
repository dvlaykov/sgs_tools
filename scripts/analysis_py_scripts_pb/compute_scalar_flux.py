# Load code libraries
import numpy as np
import pdb # to pause execution use pdb.set_trace()
import gc
import xarray as xr
import matplotlib.pyplot as plt


# Physical constants:
g = 9.81
cp = 1005.
L = 2.501E+6
Ls = 2.834E+6


def compute_scalar_flux(ds, scalar, H, z_lim=None,\
                        fluxes=True, subgrid_total=False, subgrid_total_um=False,\
                        leonard=False, resolved=False, total_flx=False, flux_grad=False,\
                        plotZgrid=False, computeLocalDiffCoef=False, cg_xy_filter=False,\
                        compute_spectra=False, updrafts_downdrafts=False):

    # Function to compute sgs-flux components for some horizontal grid resolution at some time point

    # Outline of code: 
    #                  1)  Domain and axes definitions
    #                  2)  Extract variables from dataset
    #                  3)  Correct metadata and axes of relevant fields
    #                  4)  Compute static energy
    #                  5)  Interpolation of fields
    #                  6)  Code to manually compute diffusion coefficients to aid understanding
    #                  7)  Compute spatial gradients of scalar
    #                  8)  Compute local fluxes
    #                  9) Compute non-local (counter-gradient) fluxes                                     
    #                  10) Compute resolved fluxes
    #                  11) Manually compute total vertical flux (sum of the local and nonlocal flux)
    #                  12) Compute flux gradients
    #                  13) Limit vertical extent of results for plotting
    #                  14) Pass back results to plot routine


    if flux_grad == False:
        # specific heat capacity to give manually computed
        # Smagorinsky and nonlocal terms units of W/m2 
        unitConversion = cp
    else: unitConversion = 1



    # 1)-----------------------------------------------------------------------------------------------
    # Transform coordinates into SI units to ensure correct calculation of gradients
    # For z axis I've simply converted eta into z. 
    # So eta is 'lost', but is never used anyway, and easily retrieved. 
    ds = ds.assign_coords(longitude_t=ds.longitude_t*1000, latitude_t=ds.latitude_t*1000)
    ds = ds.assign_coords(rholev_eta_rho=ds.rholev_eta_rho*H, thlev_eta_theta = ds.thlev_eta_theta*H,\
                          thlev_bl_eta_theta=ds.thlev_bl_eta_theta*H)

    dx = ds.longitude_t.values[1] - ds.longitude_t.values[0]
    Nx = len(ds.longitude_t.values)
    Ny = len(ds.latitude_t.values)

    # Vertical axis
    z_rho_ = ds.rholev_eta_rho.values
    z_th_ = ds.thlev_eta_theta.values
    z_th_bl_ = ds.thlev_bl_eta_theta.values
    #z_th_bl_ is the same as z_th but doesn't have the top point.

    # Create z-axis vectors to correct coordinate metadata for the diffusion coefficients (K) and W_1d.
    # K: shift lowest element of axis from rho level 1 to the surface. 
    # So K fields do not have data at rho level 1.
    # W_1d: prepend z=0 to z axis (add the surface level to the axis) and drop top point, 
    # keeping axis the same length. W_1d is on z_th_bl_ so ends 2 points below the top theta point.
    # F_blended needs the same correction as K.
    z_rho_sf_ = z_rho_.copy()
    z_rho_sf_[0] = 0
    z_th_sf_ = np.append(0, z_th_[:-1])
    z_th_bl_sf_ = np.append(0, z_th_bl_[:-1])

    # Find upper index of z-coordinate to impose z_lim below
    # (Axes starting from the surface with a full set of atmospheric points
    # need to reach up 1 point higher). Fields are trimmed at end of flux calculations to avoid 
    # over complicating matters.
    if z_lim is not None: idxs = np.where(z_th_ <= z_lim)
    else: idxs = np.where(z_th_ <= H+H)
    zIdx_max = np.max(idxs) + 1 # because zIdx_max is used in loop syntax range(a,b) that end at index b-1.
    z_rho = z_rho_[0:zIdx_max]
    z_th = z_th_[0:zIdx_max]
    z_th_bl = z_th_bl_[0:zIdx_max]
    z_rho_sf = z_rho_sf_[0:zIdx_max]
    z_th_sf = z_th_sf_[0:zIdx_max+1]
    z_th_bl_sf = z_th_bl_sf_[0:zIdx_max+1]

    if plotZgrid:
        # Plot to look at staggered z-grid
        fig, axes = plot_env.initialise(rows=1,cols=1, aspect_ratio=2/3.)
        fig.set_tight_layout(True)
        axes.plot(np.ones(len(z_th))[0:5],z_th[0:5], '.k', markerfacecolor='none', label=r'$z_{\theta}$')
        axes.plot(np.ones(len(z_th_bl))[0:5],z_th_bl[0:5], '^g', markerfacecolor='none', label=r'$z_{\theta\rm{ABL}}$')
        axes.plot(np.ones(len(z_rho))[0:5],z_rho[0:5], 'sb', markerfacecolor='none', label=r'$z_{\rho}$')
        axes.plot([0,2],[0,0], color='gray')

        axes.set_ylabel(r'$z$')
        axes.get_xaxis().set_visible(False)
        axes.legend(frameon=False)
        fig.savefig('../plots/v_grid.eps')
        plt.close() 

        pdb.set_trace() 


    # 2)-----------------------------------------------------------------------------------------------
    # Extract required variables from ds
    # (this isn't strictly necessary, but makes the code easier to read)
    if scalar== 'theta': 
        phi = ds['STASH_m01s16i004']
        qv = ds['STASH_m01s00i010']
        ql = ds['STASH_m01s00i254']
        qf = ds['STASH_m01s00i012']
        gamma = ds['STASH_m01s03i719']
       
        # compute virtual potential temperature, assuming
        # no liquid water present and specific humidity equal to vapour mixing ratio:
        theta = ds['STASH_m01s00i004']
        theta_v = theta.copy()
        theta_v.data = theta.data*(1 + 0.622*qv.data)

        if subgrid_total_um: 
            F_blend_grad_rho = ds['STASH_m01s03i714']
            F_blend_nongrad_rho = ds['STASH_m01s03i715']
            F_blend_entrain_rho = ds['STASH_m01s03i716']
            F_blended_rho = ds['STASH_m01s03i216']
        flux0 = ds['STASH_m01s03i217']
    if scalar == 'q': 
        phi = ds['STASH_m01s00i010']
        if subgrid_total_um: F_blended_rho = ds['']
        flux0 = ds['STASH_m01s03i234']
    
    u = ds['STASH_m01s00i002']
    v = ds['STASH_m01s00i003']
    w = ds['STASH_m01s00i150']
 
    rhoK_Ri_rho = ds['STASH_m01s03i504']         
    rhoK_sf_rho = ds['STASH_m01s03i506']         
    rhoK_sc_rho = ds['STASH_m01s03i508']
    #rhokh = ds['STASH_m01s03i717']
    rhokh = ds['STASH_m01s03i472']
    rhokhz = ds['STASH_m01s03i718']

    rho = ds['STASH_m01s00i389']
        
    W_1d_th = ds['STASH_m01s03i513']         

    z_bl = ds['STASH_m01s03i025']
    z_loc = ds['STASH_m01s03i358']

    Bflux0 = ds['STASH_m01s03i467']
    u_star = ds['STASH_m01s03i465']
    w_star = ds['STASH_m01s03i466']         

    if leonard:
        if scalar == 'theta': L_3phi_rho = ds['STASH_m01s03i556']
        if scalar == 'q': L_3phi_rho = ds['STASH_m01s03i557']
        c_L = ds['STASH_m01s03i552']

        L_3phi_rho.data = np.multiply(c_L.data, L_3phi_rho.data)
        L_3phi_rho.name = 'L_3phi'

    if computeLocalDiffCoef:
        l = ds['STASH_m01s13i193']
        Shear = ds['STASH_m01s13i192']
        f_Ri = ds['STASH_m01s03i511']

    # close xarray dataset object and remove it from memory to 
    # avoid unecessary memory use during execution of function.
    # (Any local variables will be cleared from memory automatically when the function ends.)
    ds.close()
    del ds
    gc.collect()


    # 3)-----------------------------------------------------------------------------------------------
    # This section corrects the coordinates of variables (as explained above) 
    rhoK_Ri_rho = rhoK_Ri_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
    rhoK_sf_rho = rhoK_sf_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
    rhoK_sc_rho = rhoK_sc_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
    rhokh = rhokh.assign_coords({'rholev_eta_rho': z_rho_sf_})
    rhokhz = rhokhz.assign_coords({'rholev_eta_rho': z_rho_sf_})
    W_1d_th = W_1d_th.assign_coords({'thlev_bl_eta_theta': z_th_bl_sf_})
    if subgrid_total_um: 
        F_blend_grad_rho = F_blend_grad_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
        F_blend_nongrad_rho = F_blend_nongrad_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
        F_blend_entrain_rho = F_blend_entrain_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
        F_blended_rho = F_blended_rho.assign_coords({'rholev_eta_rho': z_rho_sf_})
    if computeLocalDiffCoef: 
        f_Ri = f_Ri.assign_coords({'thlev_eta_theta': z_th_sf_})


    # 4)-----------------------------------------------------------------------------------------------
    # Compute static energy, accounting for any liquid or frozen water formation
    # n.b. Liquid and frozen water formation has negligible effect in the CBL experiment.
    if scalar == 'theta':

        nlevs,nlat,nlong=phi.shape
        tmp = np.tile(g/cp*z_th_, (nlat,nlong,1))
        gv_arr = np.moveaxis(tmp, 2, 0)
        phi.data = phi.data + gv_arr - L/cp*ql - Ls/cp*qf
        #phi.data = T_L.data + gv_arr

        #check numpy computation by (more) manually computing it:
        #for jj in range(0,nlat):
        #    for ii in range(0,nlong):
        #        phi.data[:,jj,ii] = phi.data[:,jj,ii] + g/cp*z_th - L/cp*ql[:,jj,ii] - Ls/cp*qf[:,jj,ii]
        #        phi.data[:,jj,ii] = phi.data[:,jj,ii] + g/cp*z_th


    # 5)-----------------------------------------------------------------------------------------------
    # Interpolation section
    # Although W_1d_th does have a surface value, it is meaningless and simply
    # has the array's initialisation value of 1. This surface value should be 
    # discarded to avoid it contaminating the interpolation. Because the 
    # blending weight is used together with the diffusion coefficients, it also 
    # makes sense to remove the surface data from the diffusion coefficients. 
    # Further, surface data of diffusion coefficients cannot be used for interpolating coefficients to
    # atmospheric points because it is associated with different physical processes.
    # It also turns out that the lowest common theta point for the flux calculations is level 2, 
    # so we remove theta level 1 (theta level 0 is for the surface).
    W_1d_th = W_1d_th.isel(thlev_bl_eta_theta = list(range(2,len(z_th_bl_sf_))) )
    rhokh = rhokh.isel(rholev_eta_rho = list(range(1,len(z_rho_sf_))) )
    rhokhz = rhokhz.isel(rholev_eta_rho = list(range(1,len(z_rho_sf_))) )
    rhoK_Ri_rho = rhoK_Ri_rho.isel(rholev_eta_rho = list(range(1,len(z_rho_sf_))) )
    rhoK_sf_rho = rhoK_sf_rho.isel(rholev_eta_rho = list(range(1,len(z_rho_sf_))) )
    rhoK_sc_rho = rhoK_sc_rho.isel(rholev_eta_rho = list(range(1,len(z_rho_sf_))) )


    # Calculate down-gradient eddy diffusion first on theta points, 
    # as in the UM (see ../diffusion_and_filtering/eg_diff_ctl.F90), 
    # so first interpolate required fields to theta points.
    # The 1st rho level with data for the Ks is rho level 2, so we can interpolate 
    # to theta level 2 and above (theta level numbering starts from 0 at the surface).
    # The top boundary is a theta point, which we cannot interpolate to from rho levels,
    # hence loop runs to nlevs-1 rather than nlevs. So top values of interpolated fields
    # are erroneous, but trimmed below anyway before plotting.
    rhoK_Ri_th = phi.copy()
    rhoK_sf_th = phi.copy()
    rhoK_sc_th = phi.copy()
    nlevs,nlat,nlong=phi.shape
    for kk in range(1,nlevs-1):
        rhoK_Ri_th.data[kk,:,:]=(rhoK_Ri_rho.data[kk,:,:]+rhoK_Ri_rho.data[kk-1,:,:])/2.
        rhoK_sf_th.data[kk,:,:]=(rhoK_sf_rho.data[kk,:,:]+rhoK_sf_rho.data[kk-1,:,:])/2.
        rhoK_sc_th.data[kk,:,:]=(rhoK_sc_rho.data[kk,:,:]+rhoK_sc_rho.data[kk-1,:,:])/2.
    # Remove theta level 1 data which we could not interpolate to
    rhoK_Ri_th = rhoK_Ri_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))
    rhoK_sf_th = rhoK_sf_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))
    rhoK_sc_th = rhoK_sc_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))


    # Compute blended diffusion coefficient
    rhoK_th = rhoK_Ri_th.copy()
    rhoK_th.data = np.maximum(np.multiply(W_1d_th.data, np.add(rhoK_sf_th.data, rhoK_sc_th.data)), rhoK_Ri_th.data)
    rhoK_th.name = 'rhoK_th'


    # Interpolate rhoK_th and W_1d_th to rho levels for vertical fluxes
    # The lowest point of rhoK_th and W_1d_th is theta level 2, so we 
    # can interpolate onto rho level 3 and above.
    # W_1d = W_1d_th.interp(thlev_bl_eta_theta=z_rho[2:], method=interp_method)
    # W_1d = W_1d_th.rename({'thlev_bl_eta_theta':'rholev_eta_rho'})
    W_1d_rho = rhoK_Ri_rho.copy()
    rhoK_rho = rhoK_Ri_rho.copy()
    nlevs,nlat,nlong=rhoK_Ri_rho.shape

    for kk in range(1,nlevs-3):
        W_1d_rho.data[kk,:,:]=(W_1d_th.data[kk,:,:]+W_1d_th.data[kk-1,:,:])/2.
        rhoK_rho.data[kk,:,:]=(rhoK_th.data[kk,:,:]+rhoK_th.data[kk-1,:,:])/2.
    # Remove data on rho level 2 which could not be interpolate onto
    # (n.b. the data for these new fields was stored onto copies of rhoK_Ri_rho that starts on rho level 2)
    W_1d_rho = W_1d_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))
    rhoK_rho = rhoK_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))

    # Check calculation of rhoK_rho
    #plt.figure()
    #plt.plot(rhoK_rho.data[0:10,10,10], '.k')
    #plt.plot(rhokh.data[1:10,10,10], 'sb', markersize=10, markerfacecolor='none')
    #plt.plot(rhoK_Ri_rho.data[1:10,10,10], 'or', markersize=10, markerfacecolor='none')
    #plt.show()
    #pdb.set_trace()

    # For east-west fluxes:
    # We need rhoK_th above u points
    # ../diffusion_and_filtering/eg_diff_ctl.F90 indicates that forward differencing is used

    # Interpolate rhokh to theta points
    rhoK_th = phi.copy()
    nlevs,nlat,nlong=phi.shape
    for kk in range(1,nlevs-1):
        rhoK_th.data[kk,:,:] = (rhokh.data[kk,:,:]-rhokh.data[kk-1,:,:])/2.
    rhoK_th = rhoK_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))

    # Now that we have interpolated rhokh to the theta points, we can 
    # remove the lowest point of rhokh and set rhoK_rho to rhokh
    rhokh = rhokh.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))
    rhoK_rho.data = rhokh.data

    rhoK_th_u = u.copy()
    rhoK_th_u = rhoK_th_u.isel(rholev_eta_rho = list(range(0, len(z_rho_)-3)))
    nlevs,nlat,nlong=rhoK_th_u.shape
    for ii in range(0,nlong):
        if ii <= nlong-2:
            rhoK_th_u.data[0:len(z_th_)-1,:,ii]=(rhoK_th.data[:,:,ii+1]+rhoK_th.data[:,:,ii])/2.
        if ii == nlong-1:
            rhoK_th_u.data[0:len(z_th_)-1,:,ii]=(rhoK_th.data[:,:,ii]+rhoK_th.data[:,:,ii-1])/2.
    # correct z-coordinate
    rhoK_th_u = rhoK_th_u.rename({'rholev_eta_rho':'thlev_eta_theta'})
    rhoK_th_u = rhoK_th_u.assign_coords({'thlev_eta_theta': z_th_[1:-2]})

    # For north-south fluxes:
    # We need rhoK_th above v points
    # ../diffusion_and_filtering/eg_diff_ctl.F90 indicates that forward differencing is used
    rhoK_th_v = v.copy()
    rhoK_th_v = rhoK_th_v.isel(rholev_eta_rho = list(range(0, len(z_rho_)-3)))
    nlevs,nlat,nlong=rhoK_th_v.shape
    for jj in range(0,nlat):
        if jj <= nlat-2:
            rhoK_th_v.data[0:len(z_th_)-1,jj,:]=(rhoK_th.data[:,jj+1,:]+rhoK_th.data[:,jj,:])/2.
        if jj == nlat-1:
            rhoK_th_v.data[0:len(z_th_)-1,jj,:]=(rhoK_th.data[:,jj,:]+rhoK_th.data[:,jj-1,:])/2.
    # correct z-coordinate
    rhoK_th_v = rhoK_th_v.rename({'rholev_eta_rho':'thlev_eta_theta'})
    rhoK_th_v = rhoK_th_v.assign_coords({'thlev_eta_theta': z_th_[1:-2]})


    if resolved:
        # For east-west fluxes:
        # We need phi on theta levels above the u points
        phi_u = u.copy()
        nlevs,nlat,nlong=u.shape
        for ii in range(0,nlong):
            if ii <= nlong-2:
                phi_u.data[:,:,ii]=(phi.data[:,:,ii+1]+phi.data[:,:,ii])/2.
            if ii == nlong-1:
                phi_u.data[:,:,ii]=(phi.data[:,:,ii]+phi.data[:,:,ii-1])/2.
        # correct z-coordinate
        phi_u = phi_u.rename({'rholev_eta_rho':'thlev_eta_theta'})
        phi_u = phi_u.assign_coords({'thlev_eta_theta': z_th_})

        # For north-south fluxes:
        # We need phi on theta levels above the v points
        phi_v = v.copy()
        nlevs,nlat,nlong=v.shape
        for jj in range(0,nlat):
            if jj <= nlat-2:
                phi_v.data[:,jj,:]=(phi.data[:,jj+1,:]+phi.data[:,jj,:])/2.
            if jj == nlat-1:
                phi_v.data[:,jj,:]=(phi.data[:,jj,:]+phi.data[:,jj-1,:])/2.
        # correct z-coordinate
        phi_v = phi_v.rename({'rholev_eta_rho':'thlev_eta_theta'})
        phi_v = phi_v.assign_coords({'thlev_eta_theta': z_th_})

        # vertical fluxes:
        # We need phi on rho levels below phi points
        phi_rho = rho.copy()
        nlevs,nlat,nlong=rho.shape
        for kk in range(1,nlevs):
            phi_rho.data[kk,:,:]=(phi.data[kk,:,:]+phi.data[kk-1,:,:])/2.
        # Cannot interpolate to 1st rho level
        phi_rho = phi_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_))))

        # We need w on rho levels
        # w_rho = w.interp(thlev_eta_theta=z_rho[1:], method=interp_method)
        # w_rho = w_rho.rename({'thlev_eta_theta':'rholev_eta_rho'})
        w_rho = rho.copy()
        nlevs,nlat,nlong=w_rho.shape
        for kk in range(1,nlevs):
            w_rho.data[kk,:,:]=(w.data[kk,:,:]+w.data[kk-1,:,:])/2.
        # Cannot interpolate to 1st rho level
        w_rho = w_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_))))

        #u_th = u.interp(rholev_eta_rho=z_th, method=interp_method)
        #u_th = u_th.rename({'rholev_eta_rho':'thlev_eta_theta'})

        # We need u on theta levels above u points
        u_th = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for kk in range(0,nlevs):
            if kk <= nlevs-2:
                u_th.data[kk,:,:]=(u.data[kk,:,:]+u.data[kk+1,:,:])/2.
            if kk == nlevs-1:
                u_th.data[kk,:,:]=(u.data[kk,:,:]+u.data[kk-1,:,:])/2.
        # correct (x,y)-coordinates
        u_th = u_th.rename({'longitude_t':'longitude_cu'})
        u_th = u_th.rename({'latitude_t':'latitude_cu'})
        u_th = u_th.assign_coords({'longitude_cu': u.longitude_cu.values})
        u_th = u_th.assign_coords({'latitude_cu': u.latitude_cu.values})

        # We need v on theta levels above v points
        v_th = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for kk in range(0,nlevs):
            if kk <= nlevs-2:
                v_th.data[kk,:,:]=(v.data[kk,:,:]+v.data[kk+1,:,:])/2.
            if kk == nlevs-1:
                v_th.data[kk,:,:]=(v.data[kk,:,:]+v.data[kk-1,:,:])/2.
        # correct (x,y)-coordinates
        v_th = v_th.rename({'longitude_t':'longitude_cv'})
        v_th = v_th.rename({'latitude_t':'latitude_cv'})
        v_th = v_th.assign_coords({'longitude_cv': v.longitude_cv.values})
        v_th = v_th.assign_coords({'latitude_cv': v.latitude_cv.values})

        # We need rho on theta levels above u and v points
        # 1st interpolate rho to theta levels, then interpolate to u,v points
        rho_th = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for kk in range(0,nlevs-1):
            rho_th.data[kk,:,:]=(rho.data[kk+1,:,:]+rho.data[kk,:,:])/2.

        rho_th_u = u.copy()
        nlevs,nlat,nlong=u.shape
        for ii in range(0,nlong):
            if ii <= nlong-2:
                rho_th_u.data[:,:,ii]=(rho_th.data[:,:,ii+1]+rho_th.data[:,:,ii])/2.
            if ii == nlong-1:
                rho_th_u.data[:,:,ii]=(rho_th.data[:,:,ii]+rho_th.data[:,:,ii-1])/2.
        # correct z-coordinate
        rho_th_u = rho_th_u.rename({'rholev_eta_rho':'thlev_eta_theta'})
        rho_th_u = rho_th_u.assign_coords({'thlev_eta_theta': z_th_})

        rho_th_v = v.copy()
        nlevs,nlat,nlong=v.shape
        for ii in range(0,nlong):
            if ii <= nlong-2:
                rho_th_v.data[:,:,ii]=(rho_th.data[:,:,ii+1]+rho_th.data[:,:,ii])/2.
            if ii == nlong-1:
                rho_th_v.data[:,:,ii]=(rho_th.data[:,:,ii]+rho_th.data[:,:,ii-1])/2.
        # correct z-coordinate
        rho_th_v = rho_th_v.rename({'rholev_eta_rho':'thlev_eta_theta'})
        rho_th_v = rho_th_v.assign_coords({'thlev_eta_theta': z_th_})



    # 6)-----------------------------------------------------------------------------------------------
    if computeLocalDiffCoef:
        # In the UM the diffusion coefficent is first computed using variables 
        # (mixing length, shear magnitude and functions of Ri) held on theta levels.
        # So we do the same here before interpolating result onto 
        # rho levels for comparison with the UM output rhoK_Ri also held on rho levels.
        # The manual computation now matches the UM output.

        #Trim l to align with W_1d_th
        l = l.isel(thlev_eta_theta = list(range(1,len(z_th_sf_)-2)))
        z_ = z_th_[1:-2]
        #print(z_)
        #print(l.thlev_eta_theta.values)
        #print(W_1d_th.thlev_bl_eta_theta.values)

        l_blend = l.copy()
        l_smag = l.copy()
        l_bl = l.copy()
        kappa = 0.4
        cs = 0.23
        z0 = 0.1

        nlevs,nlat,nlong=l.shape
        for jj in range(0,nlat):
            for ii in range(0,nlong):
                lambda0 = max(40.0, 0.15*z_bl.data[t_pnt,jj,ii])
                #print(lambda0)
                #l_bl.data[:,jj,ii] = kappa*z_*lambda0/(lambda0 + kappa*z_)
                l_bl.data[:,jj,ii] = kappa*(z_ + z0) / (1. + kappa*(z_ + z0)/lambda0)
                tmp = 1./(kappa*(z_ + z0))**2 + 1./(cs*dx)**2
                l_smag.data[:,jj,ii] = tmp**(-1./2)
                l_blend.data[:,jj,ii] = W_1d_th.data[:,jj,ii]*l_bl.data[:,jj,ii] +\
                                                (1. - W_1d_th.data[:,jj,ii])*l_smag.data[:,jj,ii]

        # N.b. this calculation of the mixing length is valid for points above the first atmosphere level (UMDP:024). 
        # For the 1st atmosphere point the mixing length requires a (log law) correction for the influence of the wall.
        # Since we drop the 1st level anyway, this can be ignored.

        if checkMixingL == True:
            #Look at mixing lengths:
            plt.rcParams.update({'font.size': 13})
            fig1, ax1 = plt.subplots()

            ax1.plot(l.data[0:zIdx_max,10,10], z_th_[0:zIdx_max], 'o-', color='gray', markerfacecolor='none', label=r'$l$ -- RNEUTML')
            ax1.plot(l_bl.data[0:zIdx_max,10,10], z_th_[0:zIdx_max], 'o-k', markerfacecolor='none', label=r'$l_{\rm 1D}$')
            ax1.plot(l_smag.data[0:zIdx_max,10,10], z_th_[0:zIdx_max], 'sr', markerfacecolor='none', label=r'$l_{\rm smag}$')
            ax1.plot(l_blend.data[0:zIdx_max,10,10], z_th[0:zIdx_max], '.-g', label=r'$l_{\rm blended}$')
            ax1.legend(frameon=False)
            ax1.set_xlabel(r'$l$ (m)')
            ax1.set_ylabel(r'$z$ (m)')
            plt.savefig('../plots/mixingLengths.eps')
            plt.close()


        # We can now compute the diffusion coefficient: 
        rho_th_ = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for kk in range(0,nlevs-1):
            rho_th_.data[kk,:,:]=(rho.data[kk+1,:,:]+rho.data[kk,:,:])/2.

        rho_th_ = rho_th_.isel(thlev_eta_theta = list(range(1,len(z_th_sf_)-2)))
        Shear = Shear.isel(thlev_eta_theta = list(range(1,len(z_th_sf_)-2)))
        f_Ri = f_Ri.isel(thlev_eta_theta = list(range(2,len(z_th_sf_)-1)))
        #print(rho_th_.thlev_eta_theta.values)
        #print(l_blend.thlev_eta_theta.values)
        #print(Shear.thlev_eta_theta.values)
        #print(f_Ri.thlev_eta_theta.values)
        #pdb.set_trace()

        factor = 1
        rhoK_Ri_th2 = rho_th_.copy()
        #rhoK_Ri_th2.data = rho_th_.data*factor*l.data**2*Shear.data*f_Ri.data
        #rhoK_Ri_th2.data = rho_th_.data*factor*l_smag.data**2*Shear.data*f_Ri.data
        rhoK_Ri_th2.data = rho_th_.data*factor*l_blend.data**2*Shear.data*f_Ri.data

        #print(rhoK_Ri_rho.rholev_eta_rho.values)
        #print(rhoK_Ri_th2.thlev_eta_theta.values)
        rhoK_Ri_rho2 = rhoK_Ri_rho.copy()
        nlevs,nlat,nlong = rhoK_Ri_rho.shape
        for kk in range(1,nlevs-2):
            rhoK_Ri_rho2.data[kk,:,:]=(rhoK_Ri_th2.data[kk,:,:]+rhoK_Ri_th2.data[kk-1,:,:])/2.
        rhoK_Ri_rho2 = rhoK_Ri_rho2.isel(rholev_eta_rho = list(range(1, len(z_rho_)-2)))


        # Look at diffusion coefficients
        plt.rcParams.update({'font.size': 13})
        fig2, axis1 = plt.subplots()

        print(z_rho_[0:zIdx_max])
        print(rhoK_Ri_rho.rholev_eta_rho.values)
        print(rhoK_Ri_rho2.rholev_eta_rho.values)
        #pdb.set_trace()

        axis1.plot(rhoK_Ri_rho.data[1:zIdx_max-1,10,10], z_rho_[2:zIdx_max], 'o-g', markerfacecolor='none', label=r'$rhoK_{\phi}$')
        axis1.plot(rhoK_Ri_rho2.data[0:zIdx_max-2,10,10], z_rho_[2:zIdx_max], '.-k', label=r'$rhoK_{\phi}$ -- Manual')

        xbnds = axis1.get_xbound()
        axis1.plot( [xbnds[0],xbnds[1]], [z_bl.data[t_pnt,10,10],z_bl.data[t_pnt,10,10]],\
                           ':', color='gray', linewidth=1 )

        axis1.set_ylabel(r'$z$ (m)')
        axis1.set_xlabel(r'$rhoK_{\phi}$')

        axis2 = axis1.twiny()
        axis2.plot(phi.data[0:zIdx_max,10,10], z_[0:zIdx_max], 'x-', color='red', label=r'$\theta_l$')
        axis2.set_xlabel(r'$\theta_l$ (K)')

        lines, labels = axis1.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis1.legend(lines + lines2, labels + labels2, frameon=False)
        plt.savefig('../plots/diffusionCoef.eps')
        plt.close()
        pdb.set_trace()


    # 7)-----------------------------------------------------------------------------------------------
    # We compute gradients manually to align with the UM difference stencils
    # on the Arakawa-C/Charley Phillips grid.
    # ../diffusion_and_filtering/eg_turb_smagorinsky.F90 indicates backward differencing is used.
    # This is needed to avoid introducing 'errors' in the calculation.

    # (x,y) fluxes:
    # We need theta gradients on theta levels stored above the u and v positions.
    # Method is to store gradients on u grid, before correcting the field's z coordinate
    # East-west gradients
    dphi_dx_th = u.copy()
    nlevs,nlat,nlong=u.shape
    for ii in range(0,nlong):
        if ii > 0:
            dphi_dx_th.data[:,:,ii]=(phi.data[:,:,ii]-phi.data[:,:,ii-1])/dx
        # use forward differencing at western boundary:
        if ii == 0:
            dphi_dx_th.data[:,:,ii]=(phi.data[:,:,ii+1]-phi.data[:,:,ii])/dx
    # correct z-coordinate:
    dphi_dx_th = dphi_dx_th.rename({'rholev_eta_rho':'thlev_eta_theta'})
    dphi_dx_th = dphi_dx_th.assign_coords({'thlev_eta_theta': z_th_})

    # North-south gradients 
    dphi_dy_th = v.copy()
    nlevs,nlat,nlong=v.shape
    for jj in range(0,nlat):
        if jj > 0:
            dphi_dy_th.data[:,jj,:]=(phi.data[:,jj,:]-phi.data[:,jj-1,:])/dx
        # use forward differencing at southern boundary:
        if jj == 0:
            dphi_dy_th.data[:,jj,:]=(phi.data[:,jj+1,:]-phi.data[:,jj,:])/dx
    # correct z-coordinate:
    dphi_dy_th = dphi_dy_th.rename({'rholev_eta_rho':'thlev_eta_theta'})
    dphi_dy_th = dphi_dy_th.assign_coords({'thlev_eta_theta': z_th_})

    # Vertical fluxes for scalars are on rho/mass levels
    # Compute using theta field and store on the mass levels 
    dphi_dz_rho = rho.copy()
    nlevs,nlat,nlong=rho.shape

    for kk in range(1,nlevs):
        dphi_dz_rho.data[kk,:,:]=(phi.data[kk,:,:]-phi.data[kk-1,:,:])/\
                                       (phi.thlev_eta_theta.values[kk]-phi.thlev_eta_theta.values[kk-1])
    # Remove 1st rho level where we could not calculate gradient from theta level data.
    dphi_dz_rho = dphi_dz_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_))))


    # Trim variables to align with other fields:
    dphi_dx_th = dphi_dx_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))
    dphi_dy_th = dphi_dy_th.isel(thlev_eta_theta = list(range(1, len(z_th_)-2)))
    dphi_dz_rho = dphi_dz_rho.isel(rholev_eta_rho = list(range(1, len(z_th_)-1)))
    # check that vertical gradients are on the correct levels:
    #print(dphi_dz_rho.rholev_eta_rho.values)
    #pdb.set_trace()


    # 8)------------------------------------------------------------------------------------------
    # Local (down-gradient) momentum flux:

    # Compute x-component of flux
    F_Sx_th = dphi_dx_th.copy()
    F_Sx_th.data = -np.multiply(rhoK_th_u.data, dphi_dx_th.data)*unitConversion
    F_Sx_th.name = 'F_Sx_th'

    # Compute y-component of flux
    F_Sy_th = dphi_dy_th.copy()
    F_Sy_th.data = -np.multiply(rhoK_th_v.data, dphi_dy_th.data)*unitConversion
    F_Sy_th.name = 'F_Sy_th'

    # Compute z-component of flux
    F_Sz_rho = dphi_dz_rho.copy()
    F_Sz_rho.data = -np.multiply(rhoK_rho.data, dphi_dz_rho.data)*unitConversion
    F_Sz_rho.name = 'F_Sz_rho'


    # 9)----------------------------------------------------------------------------------
    # Non-local heat flux (non-local moisture flux set to zero) on rho levels
    if scalar == 'theta':

        A_ga = 3.26
        G_max = 1E-3

        F_nl_rho = W_1d_rho.copy()
        F_nl_rho.name = 'F_nl_rho'

        rhoK_sf_rho = rhoK_sf_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))
        rhokhz = rhokhz.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))

        nlevs,nlat,nlong=W_1d_rho.shape
        for jj in range(nlat):
            for ii in range(nlong):

                w_m3 = u_star.data[jj,ii]**3 + 0.25*z_bl.data[jj,ii]*Bflux0.data[jj,ii]
 
                flux0_base_units = flux0.data[jj,ii]/unitConversion/rho.data[0,jj,ii] 

                sigma_T1 = 1.93*flux0_base_units/(w_m3**(1./3))

                gamma_th = min(A_ga*sigma_T1/z_bl.data[jj,ii], G_max)

                #F_nl_rho.data[:,jj,ii] = W_1d_rho.data[:,jj,ii]*rhoK_sf_rho.data[:,jj,ii]*gamma_th*unitConversion
                F_nl_rho.data[:,jj,ii] = W_1d_rho.data[:,jj,ii]*rhokhz.data[:,jj,ii]*gamma_th*unitConversion
                #F_nl_rho.data[:,jj,ii] = W_1d_rho.data[:,jj,ii]*rhokhz.data[:,jj,ii]*gamma.data[jj,ii]*unitConversion


    # 10)---------------------------------------------------------------------------------- 
    if resolved:
        
        rho = rho.isel(rholev_eta_rho = list(range(1, len(z_rho_))))

        # x resolved turbulent flux: 
        mean_u = u_th.mean(dim=['longitude_cu','latitude_cu'])
        tmp = np.tile(mean_u, (Nx,Ny,1))
        mean_u_rep = np.moveaxis(tmp, 2, 0)

        mean_phi_u = phi_u.mean(dim=['longitude_cu','latitude_cu'])
        tmp = np.tile(mean_phi_u, (Nx,Ny,1))
        mean_phi_u_rep = np.moveaxis(tmp, 2, 0)

        F_res_x = phi.copy()
        F_res_x.data = np.multiply(u_th.data-mean_u_rep,phi_u.data-mean_phi_u_rep)*rho_th_u.data*unitConversion
        F_res_x.name = 'F_res_x'
   
        # y resolved turbulent flux: 
        mean_v = v_th.mean(dim=['longitude_cv','latitude_cv'])
        tmp = np.tile(mean_v, (Nx,Ny,1))
        mean_v_rep = np.moveaxis(tmp, 2, 0)

        mean_phi_v = phi_v.mean(dim=['longitude_cv','latitude_cv'])
        tmp = np.tile(mean_phi_v, (Nx,Ny,1))
        mean_phi_v_rep = np.moveaxis(tmp, 2, 0)

        F_res_y = phi.copy()
        F_res_y.data = np.multiply(v_th.data-mean_v_rep,phi_v.data-mean_phi_v_rep)*rho_th_v.data*unitConversion
        F_res_y.name = 'F_res_y'

        # z resolved turbulent flux: 
        mean_w = w_rho.mean(dim=['longitude_t','latitude_t'])
        tmp = np.tile(mean_w, (Nx,Ny,1))
        mean_w_rep = np.moveaxis(tmp, 2, 0)

        mean_phi_rho = phi_rho.mean(dim=['longitude_t','latitude_t'])
        tmp = np.tile(mean_phi_rho, (Nx,Ny,1))
        mean_phi_rho_rep = np.moveaxis(tmp, 2, 0)

        F_res_z = phi_rho.copy()
        F_res_z.data = np.multiply(w_rho.data-mean_w_rep,phi_rho.data-mean_phi_rho_rep)*rho.data*unitConversion
        F_res_z.name = 'F_res_z'


    # 11)---------------------------------------------------------------------------------- 
    if subgrid_total:
        sg_total = F_nl_rho.copy()
        sg_total.data = F_Sz_rho.data + F_nl_rho.data 
        sg_total.name = 'sg_total'

    if subgrid_total_um:
        F_blend_expl_rho = F_blended_rho.copy()
        #F_blend_expl_rho.data = F_blend_grad_rho.data + F_blend_nongrad_rho.data + F_blend_entrain_rho.data 
        F_blend_expl_rho.data = F_blend_grad_rho.data + F_blend_nongrad_rho.data

    # Trim blended and leonard terms to align with other fields
    if subgrid_total_um:
        F_blend_grad_rho = F_blend_grad_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_sf_))))
        F_blend_nongrad_rho = F_blend_nongrad_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_sf_))))
        F_blend_entrain_rho = F_blend_entrain_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_sf_))))
        F_blended_rho = F_blended_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_sf_))))
        F_blend_expl_rho = F_blend_expl_rho.isel(rholev_eta_rho = list(range(1, len(z_rho_sf_))))
    if leonard: 
        c_L = c_L.isel(rholev_eta_rho = list(range(2, len(z_rho_))))
        L_3phi_rho = L_3phi_rho.isel(rholev_eta_rho = list(range(2, len(z_rho_))))

    if total_flx:
       F_res_z = F_res_z.isel(rholev_eta_rho = list(range(1, len(z_rho_)-1)))
       total = F_nl_rho.copy()

       total.data = F_res_z.data + sg_total.data + L_3phi_rho.data
       total.name = 'total flux'


    # 12)---------------------------------------------------------------------------------- 
    if flux_grad: #NOT COMPLETE

        # Turbulent diffusion coefficients 
        # come multipled by density, so here divide by density to get correct/standard units for flux gradients.
        # This doesn't have any great practical effect because density is close to unity 
        # and the fluid in the lower atmosphere is nearly incompressible.

        # Local flux gradients
        grad_F_Sx = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for ii in range(0,nlong):
            if ii > 0:
                grad_F_Sx[:,:,ii] = (F_Sx_th[:,:,ii] - F_Sx_th[:,:,ii-1])/dx
            if ii == 0:
                grad_F_Sx[:,:,ii] = (F_Sx_th[:,:,ii+1] - F_Sx_th[:,:,ii])/dx

        grad_F_Sy = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for jj in range(0,nlat):
            if jj > 0:
                grad_F_Sy[:,jj,:] = (F_Sy_th[:,jj,:] - F_Sy_th[:,jj-1,:])/dx
            if jj == 0:
                grad_F_Sy[:,jj,:] = (F_Sy_th[:,jj+1,:] - F_Sy_th[:,jj,:])/dx

        grad_F_Sz = phi.copy()
        nlevs,nlat,nlong=phi.shape
        for kk in range(0,nlevs):
            if jj > 0:
                grad_F_Sz[:,jj,:] = (F_Sz_th[:,jj,:] - F_Sz_th[:,jj-1,:])/dx
            if jj == 0:
                grad_F_Sz[:,jj,:] = (F_Sz_th[:,jj+1,:] - F_Sz_th[:,jj,:])/dx

        
        # Nonlocal flux gradients

        if leonard:
            L_3phi_rho.data = np.divide(L_3phi_rho.data,rho.data)



    # 13)---------------------------------------------------------------------------------- 
    # Limit top height of atmospheric fields for plotting:
    F_Sx_th = F_Sx_th.isel(thlev_eta_theta = np.arange(zIdx_max))
    F_Sy_th = F_Sy_th.isel(thlev_eta_theta = np.arange(zIdx_max))
    F_Sz_rho = F_Sz_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
    F_nl_rho = F_nl_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
    if subgrid_total:
        sg_total = sg_total.isel(rholev_eta_rho = np.arange(zIdx_max))
    if subgrid_total_um:
        F_blend_grad_rho = F_blend_grad_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
        F_blend_nongrad_rho = F_blend_nongrad_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
        F_blend_entrain_rho = F_blend_entrain_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
        F_blend_expl_rho = F_blend_expl_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
        F_blended_rho = F_blended_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
    if leonard:
        L_3phi_rho = L_3phi_rho.isel(rholev_eta_rho = np.arange(zIdx_max))
    if resolved:
        F_res_x = F_res_x.isel(thlev_eta_theta = np.arange(zIdx_max))
        F_res_y = F_res_y.isel(thlev_eta_theta = np.arange(zIdx_max))
        F_res_z = F_res_z.isel(rholev_eta_rho = np.arange(zIdx_max))
    if total_flx:
        total = total.isel(rholev_eta_rho = np.arange(zIdx_max))
    if cg_xy_filter:       
        u_th = u_th.isel(thlev_eta_theta = np.arange(zIdx_max))
        v_th = v_th.isel(thlev_eta_theta = np.arange(zIdx_max))
        u = u.isel(rholev_eta_rho= np.arange(zIdx_max))
        v = v.isel(rholev_eta_rho= np.arange(zIdx_max))
        w_rho = w_rho.isel(rholev_eta_rho= np.arange(zIdx_max))
        phi_u = phi_u.isel(thlev_eta_theta= np.arange(zIdx_max))
        phi_v = phi_v.isel(thlev_eta_theta= np.arange(zIdx_max))
        phi_rho = phi_rho.isel(rholev_eta_rho= np.arange(zIdx_max))
        rho_th_u = rho_th_u.isel(thlev_eta_theta= np.arange(zIdx_max))
        rho_th_v = rho_th_v.isel(thlev_eta_theta= np.arange(zIdx_max))
        rho = rho.isel(rholev_eta_rho= np.arange(zIdx_max))
        c_L = c_L.isel(rholev_eta_rho= np.arange(zIdx_max))
    if compute_spectra:
        w = w.isel(thlev_eta_theta= np.arange(zIdx_max))
        phi = phi.isel(thlev_eta_theta= np.arange(zIdx_max))
    if updrafts_downdrafts:    
        w = w.isel(thlev_eta_theta= np.arange(zIdx_max))


    # 14)---------------------------------------------------------------------------------- 
    # Construct output by merging required data arrays into a new Xarray dataset object
    if fluxes:
        #z_bl = z_bl.isel(min15T0=t_pnt)
        z_bl.name = 'z_bl'
        ds_out = xr.merge([z_bl, F_Sx_th, F_Sy_th, F_Sz_rho], compat='override', join='override')
        if scalar == 'theta': 
            ds_out = ds_out.assign(F_nl_rho=F_nl_rho)
        if subgrid_total: ds_out = ds_out.assign(sg_total=sg_total)
        if subgrid_total_um:
            F_blend_expl_rho.name = 'F_blend_expl_rho' 
            F_blend_grad_rho.name = 'F_blend_grad_rho' 
            F_blend_nongrad_rho.name = 'F_blend_nongrad_rho' 
            ds_out = ds_out.assign(F_blend_expl_rho=F_blend_expl_rho)
            ds_out = ds_out.assign(F_blend_grad_rho=F_blend_grad_rho)
            ds_out = ds_out.assign(F_blend_nongrad_rho=F_blend_nongrad_rho)
        if leonard:
            ds_out = ds_out.assign(L_3phi_rho=L_3phi_rho)
        if resolved: 
            ds_out = ds_out.assign(F_res_x=F_res_x)
            ds_out = ds_out.assign(F_res_y=F_res_y)
            ds_out = ds_out.assign(F_res_z=F_res_z)
        if total_flx: ds_out = ds_out.assign(total=total)
    if flux_grad: 
        ds_out = ds_out.assign(F_grad_phi=F_grad_phi)
    if cg_xy_filter:
        ds_out = ds_out.assign(u_th=u_th)
        ds_out = ds_out.assign(v_th=v_th)
        ds_out = ds_out.assign(w_rho=w_rho)
        ds_out = ds_out.assign(phi_u=phi_u)
        ds_out = ds_out.assign(phi_v=phi_v)
        ds_out = ds_out.assign(phi_rho=phi_rho)
        ds_out = ds_out.assign(rho_th_u=rho_th_u)
        ds_out = ds_out.assign(rho_th_v=rho_th_v)
        ds_out = ds_out.assign(rho=rho)
        ds_out = ds_out.assign(u=u)
        ds_out = ds_out.assign(v=v)
        ds_out = ds_out.assign(c_L=c_L)
    if compute_spectra:
        ds_out = ds_out.assign(w=w)
        ds_out = ds_out.assign(phi=phi)
    if updrafts_downdrafts:
        ds_out = ds_out.assign(w=w)


    return ds_out

