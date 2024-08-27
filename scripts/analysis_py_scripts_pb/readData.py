#Functions to UM pp files 
#First created: December 2023
#Paul Burns


#load required libraries:
import numpy as np
import iris
#from iris.experimental.equalise_cubes import equalise_attributes
#from iris.util import unify_time_units
import iris.coords as icoords
import netCDF4 as nc
import xarray as xarr
import re
import pdb #to pause execution use pdb.set_trace()







def readIRIS(fnms, var=None):

    cube_list = iris.load_raw(fnms, var)
    #equalise_attributes(cube_list)
    tmp = cube_list.concatenate()
    cubeList = tmp.merge()

    return cubeList




def readXARR(datadir, suite, dxs, field_type, z_grid_type):
    ds = []

    for dx in dxs:
        fdir = f'{datadir}{suite}_{dx}_{z_grid_type}_p{field_type}000.nc'
        d = xarr.open_dataset(fdir)
        ds.append(d)

    return ds




def readXARR_t(datadir, suite, field_type, z_grid_type, dxs, dxs_all, time, dt_all, timeUnit):
    ds = []

    for i, dx_str in enumerate(dxs):
        fdir = f'{datadir}{suite}_{dx_str}_{z_grid_type}_p{field_type}000.nc'
        d = xarr.open_dataset(fdir)

        # Find time index for some time and dx
        dx_int = int(re.findall(r'\d+', dx_str)[0])
        if dx_int == 1: dx_int = 1000
        if dx_int == 10: dx_int = 10000
        idx_dt = np.where(np.array(dxs_all)==dx_int)[0][0]
        dt = dt_all[idx_dt]
        t_pnt = int(time/dt)

        #print('time index: ', t_pnt)

        if timeUnit == 'timesteps': ds.append(d.isel(min15T0=t_pnt))
        if timeUnit == 'minutes': ds.append(d.isel(min15T0_0=t_pnt, min15T0=t_pnt))

    return ds



if __name__ == '__main__':
    main()
