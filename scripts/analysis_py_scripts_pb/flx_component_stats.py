# Program to compute mean and std from a random pick of vertical profiles
# within updraft and downdraft regions
# First created: April 2024
# Author: Paul Burns



import xarray as xr
import numpy as np
import pdb


def compute_flx_component_stats(dat_in, dxs, geostrophic, cg_xy_filter, n):

    dat = dat_in.copy()
    dat_mean = dat_in.copy()
    dat_std = dat_in.copy()
    dat_out = dat_in.copy()

    for i, dx in enumerate(dxs):

        # read in random pick for some dx
        dx = int(dat[i].longitude_t.values[1] - dat[i].longitude_t.values[0])
        fnm_common = str(n) + '_' + str(dx) + '_' + geostrophic
        if cg_xy_filter and dx!=100: fnm_common = fnm_common + '_filtered'

        xIdxs = np.load('../output/updrafts/updrafts_rdm_xIdxs_' + fnm_common + '.npy', allow_pickle=False)
        yIdxs = np.load('../output/updrafts/updrafts_rdm_yIdxs_' + fnm_common + '.npy', allow_pickle=False)

        # take subset of spatial points corresponding to random pick
        dat[i] = dat[i].isel(latitude_t=yIdxs, longitude_t=xIdxs)
        dat[i] = dat[i].isel(latitude_cu=yIdxs, longitude_cu=xIdxs)
        dat[i] = dat[i].isel(latitude_cv=yIdxs, longitude_cv=xIdxs)

        # find mean and std of random pick of points
        dat_mean[i] = dat[i].mean(dim=['longitude_t','latitude_t','longitude_cu','latitude_cu','longitude_cv','latitude_cv'], skipna=True)
        dat_std[i] = dat[i].std(dim=['longitude_t','latitude_t','longitude_cu','latitude_cu','longitude_cv','latitude_cv'], skipna=True)

        #n.b. some mean/std calculations induce a warning message, which is due to nan 
        #in some fields but not in the fields of interest:
        #for key in dat[i].keys(): print(key, np.max(dat[i][key].data) )

        # using list comprehension to create new lists of names needed
        # to merge Xarray datasets below
        names = list(dat[i].keys())
        names_mean = [j + '_mean' for j in names]
        names_std = [j + '_std' for j in names]

        # use dictionary comprehension to create dictionarys
        dict_mean = {names[j]: names_mean[j] for j in range(len(names))}
        dict_std = {names[j]: names_std[j] for j in range(len(names))}

        # rename variables in mean and std datasets so that the below merge works
        dat_mean[i] = dat_mean[i].rename_vars(name_dict=dict_mean)
        dat_std[i] = dat_std[i].rename_vars(name_dict=dict_std)

        # merge new mean and std datasets with the original
        dat_out[i] = xr.merge([dat_in[i],dat_mean[i],dat_std[i]])


    return dat_out
