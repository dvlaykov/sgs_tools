# Program to find updraft and downdraft regions and to take a random pick
# of locations within those regions
# First created: April 2024
# Author: Paul Burns



import numpy as np
import random
import numbers
import pdb



def compute_up_down_drafts(dat, dxs, geostrophic, cg_xy_filter,\
                           z_level=None, randomPick=True, n=30, w2f_randomPick=False):

    baseDir = '../output/updrafts/'

    for i, dx_str in enumerate(dxs):

        dx = dat[i].longitude_t.values[1] - dat[i].longitude_t.values[0]

        #var = 'total'
        var = 'w'
        if cg_xy_filter: var = var + '_' + str(int(dx))
        field = dat[i][var].copy()

        if z_level==None:
            if var=='total' or var=='total_'+str(int(dx)): field = field.mean(dim=['rholev_eta_rho'], skipna=True)
            if var=='w' or var=='w_'+str(int(dx)): field = field.mean(dim=['thlev_eta_theta'], skipna=True)
        if isinstance(z_level, numbers.Number):
            idxs = np.where(field.rholev_eta_rho.values <= z_level)
            zIdx = np.max(idxs)
            field = field.isel(rholev_eta_rho=zIdx)
        if z_level == 'z_abl':
            z_bl_mean = dat['z_bl'].mean(dim=['longitude_t','latitude_t']).values
            print(z_bl_mean)
            idxs = np.where(field.rholev_eta_rho.values <= z_bl_mean)
            zIdx = np.max(idxs)
            field = field.isel(rholev_eta_rho=zIdx)

        updrafts_01 = field.copy()
        downdrafts_01 = field.copy()
        updrafts_01.data = field.data > 0
        downdrafts_01.data = field.data < 0
        updrafts_01.name = 'updrafts'
        downdrafts_01.name = 'downdrafts'

        dat[i] = dat[i].assign(updrafts=updrafts_01)
        dat[i] = dat[i].assign(downdrafts=downdrafts_01)

        if randomPick:
            # Find tuples of coordinates of updrafts and downdrafts
            updrafts = np.where(updrafts_01.data == True)
            downdrafts = np.where(downdrafts_01.data == True)

            # Make a random selection of n numbers
            # An index/position in updrafts[0] corresponds to the same index/position
            # in updrafts[1] (to give an (x,y) coordinate). In another way, length of updrafts[0]
            # equals the length of updrafts[1]
            updrafts_rdm_nums = random.sample(range(0, len(updrafts[0])), n)
            downdrafts_rdm_nums = random.sample(range(0, len(downdrafts[0])), n)

            # Find random indexes and coordinates 
            updrafts_rdm_xIdxs = updrafts[1][updrafts_rdm_nums]
            updrafts_rdm_yIdxs = updrafts[0][updrafts_rdm_nums]
            updrafts_rdm_x = updrafts_rdm_xIdxs*dx
            updrafts_rdm_y = updrafts_rdm_yIdxs*dx

            downdrafts_rdm_xIdxs = downdrafts[1][downdrafts_rdm_nums]
            downdrafts_rdm_yIdxs = downdrafts[0][downdrafts_rdm_nums]
            downdrafts_rdm_x = downdrafts_rdm_xIdxs*dx
            downdrafts_rdm_y = downdrafts_rdm_yIdxs*dx

            #print(updrafts_rdm_xIdxs) 
            #print(updrafts_rdm_yIdxs) 
            #print(updrafts_rdm_x) 
            #print(updrafts_rdm_y) 

            if w2f_randomPick:
                
                fnm_common = str(n) + '_' + str(int(dx)) + '_' + geostrophic
                if cg_xy_filter: fnm_common = fnm_common + '_filtered' 
                 
                np.save(baseDir + 'updrafts_rdm_xIdxs_' + fnm_common + '.npy',\
                        updrafts_rdm_xIdxs, allow_pickle=False)
                np.save(baseDir + 'updrafts_rdm_yIdxs_' + fnm_common + '.npy',\
                        updrafts_rdm_yIdxs, allow_pickle=False)

                np.save(baseDir + 'downdrafts_rdm_xIdxs_' + fnm_common + '.npy',\
                        downdrafts_rdm_xIdxs, allow_pickle=False)
                np.save(baseDir + 'downdrafts_rdm_yIdxs_' + fnm_common + '.npy',\
                        downdrafts_rdm_yIdxs, allow_pickle=False)

                np.save(baseDir + 'updrafts_rdm_nums_' + fnm_common + '.npy',\
                        updrafts_rdm_nums, allow_pickle=False)

                np.save(baseDir + 'downdrafts_rdm_nums_' + fnm_common + '.npy',\
                        downdrafts_rdm_nums, allow_pickle=False)


    return dat

