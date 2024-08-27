# Program to coarse-grain results
# First created: May 2024
# Author: Paul Burns




import numpy as np
import xarray as xr
import pdb



def cg_xy_box_filter(dat_in, eff_res_fact):

    cp = 1005.

    # Make sure reference dataset is the LES resolution
    dx = int(dat_in[0].longitude_t.values[1] - dat_in[0].longitude_t.values[0])
    if dx != 100: 
        print('Error! You need to coarsen from the LES 100-m resolution.')
        pdb.set_trace()

    # Get coarse graining factors for effective resolutions
    factors = []
    for i in range(0, len(dat_in)):
        factor = int((dat_in[i].longitude_t.values[1] - dat_in[i].longitude_t.values[0])/dx)
        factors.append(factor)

    factors = [val * eff_res_fact for val in factors]

    # Initialise Xarray containers/objects to hold results
    # Initialise new flux data (on LES mesh) in Xarray data arrays
    dat_les = dat_in[0].copy()
    tmp = dat_les['phi_u'].copy()
    tmp.name='sg_x'
    dat_les = dat_les.assign(sg_x=tmp)
    tmp = dat_les['phi_v'].copy()
    tmp.name='sg_y'
    dat_les = dat_les.assign(sg_y=tmp)
    tmp = dat_les['phi_rho'].copy()
    tmp.name='sg_total'
    dat_les = dat_les.assign(sg_z=tmp)
    
    # Create list of Xarray datasets with same length as original (for different dx)
    # Add the LES dataset to each element of list, after renaming variables. 
    # Below, list elements/datasets will be coarse grained.
    dat = []
    names = list(dat_les.keys())
    for i in range(len(dat_in)):
        dx = int(dat_in[i].longitude_t.values[1] - dat_in[i].longitude_t.values[0])*eff_res_fact
        names_new = [j + '_' + str(dx) for j in names]
        dict_new = {names[j]: names_new[j] for j in range(len(names))}
        ds_i = dat_les.rename(name_dict=dict_new)
        dat.append(ds_i)

    # Temporary arrays for computation
    sg_x = dat_in[0]['u_th'].copy()
    sg_y = dat_in[0]['v_th'].copy()
    sg_z = dat_in[0]['w_rho'].copy()
    sg_x.name = 'sg_x'
    sg_y.name = 'sg_y'
    sg_z.name = 'sg_total'

    if eff_res_fact != 1: idx0=0
    else: idx0 = 1
    for i in range(idx0,len(factors)):

        c = factors[i]
        print(i, c)
 
        dx = int(dat_in[i].longitude_t.values[1] - dat_in[i].longitude_t.values[0])*eff_res_fact

        # Compute values for coarser grid by box averaging values on LES mesh
        # n.b. this is not a rolling mean.
        pnts = c
        means_u = dat_in[0].coarsen(longitude_cu = pnts, boundary='trim').mean(skipna=True)\
                          .coarsen(latitude_cu = pnts, boundary='trim').mean(skipna=True)
        means_v = dat_in[0].coarsen(longitude_cv = pnts, boundary='trim').mean(skipna=True)\
                          .coarsen(latitude_cv = pnts, boundary='trim').mean(skipna=True)
        means_th = dat_in[0].coarsen(longitude_t = pnts, boundary='trim').mean(skipna=True)\
                          .coarsen(latitude_t = pnts, boundary='trim').mean(skipna=True)


        # Compute turbulent fluctuations that are resolved by the LES but subgrid 
        # for a coarser grid. This involves subtracting the mean values found 
        # above from the LES fields. We do this for all vertical levels at once.
        # x,y fluxes have a difference lat-lon grid to the z fluxes, so they are 
        # computed separately. 
        if c != 1:
            for ii in range(len(means_u.longitude_cu.values)):
                for jj in range(len(means_u.latitude_cu.values)):

                    # We need a turbulent quantity at every point on the LES mesh, 
                    # so here repeat the mean value for a coarse grid box across the LES mesh tile
                    tmp = np.tile(means_u['u_th'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_u = np.moveaxis(tmp, 2, 0)
                    tmp = np.tile(means_v['v_th'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_v = np.moveaxis(tmp, 2, 0)
                    tmp = np.tile(means_u['phi_u'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_phi_u = np.moveaxis(tmp, 2, 0)
                    tmp = np.tile(means_v['phi_v'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_phi_v = np.moveaxis(tmp, 2, 0)

                    # Find all LES mesh points within each coarser mesh box
                    xIdxS = pnts*ii
                    xIdxE = pnts*ii+pnts
                    yIdxS = pnts*jj
                    yIdxE = pnts*jj+pnts

                    # Compute turbulent values
                    primes_u = dat_in[0]['u_th'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_u
                    primes_v = dat_in[0]['v_th'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_v
                    primes_phi_u = dat_in[0]['phi_u'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_phi_u
                    primes_phi_v = dat_in[0]['phi_v'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_phi_v

                    # Compute fluxes resolved by LES (but subgrid for coarser mesh) and add on the subgrid LES flux
                    # to find total subgrid fluxes on coarser mesh. 
                    # We also convert units to W/m^2
                    rho_th_u = dat_in[0]['rho_th_u'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]
                    rho_th_v = dat_in[0]['rho_th_v'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]

                    sg_x.data[:,yIdxS:yIdxE,xIdxS:xIdxE] = primes_u*primes_phi_u*rho_th_u*cp +\
                                                                 dat_in[0]['F_Sx_th'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]
                    sg_y.data[:,yIdxS:yIdxE,xIdxS:xIdxE] = primes_v*primes_phi_v*rho_th_v*cp +\
                                                                 dat_in[0]['F_Sy_th'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]

            for ii in range(len(means_th.longitude_t.values)):
                for jj in range(len(means_th.latitude_t.values)):

                    tmp = np.tile(means_th['w_rho'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_w = np.moveaxis(tmp, 2, 0)
                    tmp = np.tile(means_th['phi_rho'].data[:,jj,ii], (pnts,pnts,1))
                    clmn_phi_rho = np.moveaxis(tmp, 2, 0)

                    # Find all LES mesh points within each coarser mesh box
                    xIdxS = pnts*ii
                    xIdxE = pnts*ii+pnts
                    yIdxS = pnts*jj
                    yIdxE = pnts*jj+pnts

                    primes_w = dat_in[0]['w_rho'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_w
                    primes_phi_rho = dat_in[0]['phi_rho'].data[:,yIdxS:yIdxE,xIdxS:xIdxE] - clmn_phi_rho

                    rho = dat_in[0]['rho'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]

                    sg_z.data[:,yIdxS:yIdxE,xIdxS:xIdxE] = primes_w*primes_phi_rho*rho*cp +\
                                                                 dat_in[0]['sg_total'].data[:,yIdxS:yIdxE,xIdxS:xIdxE]

        # Add new flux data (on LES mesh) to Xarray data array
        dat[i]['sg_x_' + str(dx)].data = sg_x.data 
        dat[i]['sg_y_' + str(dx)].data = sg_y.data 
        dat[i]['sg_total_' + str(dx)].data = sg_z.data 

        # Finally, compute averages across coarse grain boxes to find the results
        # (For any field that isn't a subfilter flux, this is simply a box filter operation)
        dat[i] = dat[i].coarsen(longitude_cu = pnts, boundary='trim').mean(skipna=True)\
                             .coarsen(latitude_cu = pnts, boundary='trim').mean(skipna=True)

        dat[i] = dat[i].coarsen(longitude_cv = pnts, boundary='trim').mean(skipna=True)\
                             .coarsen(latitude_cv = pnts, boundary='trim').mean(skipna=True)

        dat[i] = dat[i].coarsen(longitude_t = pnts, boundary='trim').mean(skipna=True)\
                             .coarsen(latitude_t = pnts, boundary='trim').mean(skipna=True)


        # Compute total vertical turbulent flux (the sum of resolved, subgrid and leonard terms)

        # First compute coarse-grain resolved fluxes and add to dat by overwriting existing fields
        # Compute domain means to subtract from coarse-grained fields to find turbulent resolved flux
        domain_mean_u = means_u['u_th'].mean(dim=['longitude_cu','latitude_cu'])
        domain_mean_phi_u = means_u['phi_u'].mean(dim=['longitude_cu','latitude_cu'])
        domain_mean_v = means_u['v_th'].mean(dim=['longitude_cv','latitude_cv'])
        domain_mean_phi_v = means_u['phi_v'].mean(dim=['longitude_cv','latitude_cv'])
        domain_mean_w = means_u['w_rho'].mean(dim=['longitude_t','latitude_t'])
        domain_mean_phi_rho = means_u['phi_rho'].mean(dim=['longitude_t','latitude_t'])

        Nx = len(dat[i].longitude_t.values)
        Ny = len(dat[i].latitude_t.values)
 
        tmp = np.tile(domain_mean_u, (Nx,Ny,1))
        mean_u_rep = np.moveaxis(tmp, 2, 0)

        tmp = np.tile(domain_mean_phi_u, (Nx,Ny,1))
        mean_phi_u_rep = np.moveaxis(tmp, 2, 0)

        tmp = np.tile(domain_mean_v, (Nx,Ny,1))
        mean_v_rep = np.moveaxis(tmp, 2, 0)

        tmp = np.tile(domain_mean_phi_v, (Nx,Ny,1))
        mean_phi_v_rep = np.moveaxis(tmp, 2, 0)

        tmp = np.tile(domain_mean_w, (Nx,Ny,1))
        mean_w_rep = np.moveaxis(tmp, 2, 0)

        tmp = np.tile(domain_mean_phi_rho, (Nx,Ny,1))
        mean_phi_rho_rep = np.moveaxis(tmp, 2, 0)

        dat[i]['F_res_x' + '_' + str(dx)].data = (means_u['u_th'].data-mean_u_rep) * (means_u['phi_u'].data-mean_phi_u_rep) *\
                                                 means_u['rho_th_u'].data * cp
        dat[i]['F_res_y' + '_' + str(dx)].data = (means_v['v_th'].data-mean_v_rep) * (means_v['phi_v'].data-mean_phi_v_rep) *\
                                                 means_v['rho_th_v'].data * cp
        dat[i]['F_res_z' + '_' + str(dx)].data = (means_th['w_rho'].data-mean_w_rep) * (means_th['phi_rho'].data-mean_phi_rho_rep) *\
                                                 means_th['rho'].data * cp

        # Compute Leonard term
        # 1) compute gradients using data on rho points and store on u/v points
        # 2) interpolate gradients on u/v points to rho points for vertical fluxes
       
        # 1)
        dwdx = means_u['u'].copy()
        dphi_dx = means_u['u'].copy()
        nlevs,nlat,nlong=dwdx.shape
        for ii in range(0,nlong):
            if ii > 0:
                dwdx.data[:,:,ii]=(means_th['w_rho'].data[:,:,ii]-means_th['w_rho'].data[:,:,ii-1])/dx
                dphi_dx.data[:,:,ii]=(means_th['phi_rho'].data[:,:,ii]-means_th['phi_rho'].data[:,:,ii-1])/dx
            # use forward differencing at western boundary:
            if ii == 0:
                dwdx.data[:,:,ii]=(means_th['w_rho'].data[:,:,ii+1]-means_th['w_rho'].data[:,:,ii])/dx
                dphi_dx.data[:,:,ii]=(means_th['phi_rho'].data[:,:,ii+1]-means_th['phi_rho'].data[:,:,ii])/dx
       
        dwdy = means_v['v'].copy()
        dphi_dy = means_v['v'].copy()
        nlevs,nlat,nlong=dwdx.shape
        for jj in range(0,nlong):
            if jj > 0:
                dwdy.data[:,jj,:]=(means_th['w_rho'].data[:,jj,:]-means_th['w_rho'].data[:,jj-1,:])/dx
                dphi_dy.data[:,jj,:]=(means_th['phi_rho'].data[:,jj,:]-means_th['phi_rho'].data[:,jj-1,:])/dx
            # use forward differencing at western boundary:
            if jj == 0:
                dwdy.data[:,jj,:]=(means_th['w_rho'].data[:,jj+1,:]-means_th['w_rho'].data[:,jj,:])/dx
                dphi_dy.data[:,jj,:]=(means_th['phi_rho'].data[:,jj+1,:]-means_th['phi_rho'].data[:,jj,:])/dx

        # 2) interpolate gradient products to rho points for vertical fluxes
        dwdx_dphi_dx = dwdx.copy()
        dwdx_dphi_dx.data = dwdx.data*dphi_dx.data

        dwdy_dphi_dy = dwdy.copy()
        dwdy_dphi_dy.data = dwdy.data*dphi_dy.data

        dwdx_dphi_dx_rho = means_th['rho'].copy()
        nlevs,nlat,nlong=dwdx_dphi_dx_rho.shape
        for ii in range(0,nlong):
            if ii <= nlong-2:
                dwdx_dphi_dx_rho.data[:,:,ii]=(dwdx_dphi_dx.data[:,:,ii+1]+dwdx_dphi_dx.data[:,:,ii])/2.
            if ii == nlong-1:
                dwdx_dphi_dx_rho.data[:,:,ii]=(dwdx_dphi_dx.data[:,:,ii]+dwdx_dphi_dx.data[:,:,ii-1])/2.

        dwdy_dphi_dy_rho = means_th['rho'].copy()
        nlevs,nlat,nlong=dwdy_dphi_dy_rho.shape
        for jj in range(0,nlat):
            if jj <= nlat-2:
                dwdy_dphi_dy_rho.data[:,jj,:]=(dwdy_dphi_dy.data[:,jj+1,:]+dwdy_dphi_dy.data[:,jj,:])/2.
            if jj == nlat-1:
                dwdy_dphi_dy_rho.data[:,jj,:]=(dwdy_dphi_dy.data[:,jj,:]+dwdy_dphi_dy.data[:,jj-1,:])/2.

        # Compute Leonard vertical flux and add it to dataset by over-writing original field
        L_3phi_rho = means_th['c_L'].copy()
        L_3phi_rho.data = means_th['c_L'].data*(dwdx_dphi_dx_rho.data + dwdy_dphi_dy_rho.data)
        dat[i]['L_3phi_rho' + '_' + str(dx)].data = L_3phi_rho.data

        # Compute coarse-grain total and overwrite the existing field
        dat[i]['total' + '_' + str(dx)].data = dat[i]['F_res_z' + '_' + str(dx)].data +\
                                               dat[i]['sg_total' + '_' + str(dx)].data +\
                                               dat[i]['L_3phi_rho' + '_' + str(dx)].data

        # Gross error check
        #diff = dat[i]['total' + '_' + str(dx)].data - means_th['total'].data
        #print(diff)


    return dat 



def xy_box_filter(dat, eff_res_fact):

    for i in range(len(dat)):
        dat[i] = dat[i].coarsen(longitude_t = eff_res_fact, boundary='trim').mean(skipna=True)\
                             .coarsen(latitude_t = eff_res_fact, boundary='trim').mean(skipna=True)

    return dat

