import xarray as xr
import numpy as np
from functools import partial

import utility_functions as uf


# NETCDF FOR EACH VARIABLE
netcdfs = []


def _preprocess_1_day_bounds(x, lon_bnds, lat_bnds, year, month=1, day=1):

    # Remove unwanted data by coordinates
    ds = x.where((x.latitude<=lat_bnds[1]) & (x.latitude>=lat_bnds[0]) & (x.longitude<=lon_bnds[1]) & (x.longitude>=lon_bnds[0]), drop=True)
    
    # 1 day of data to get all coordinates
    data = ds.where((ds['time.year']==year) & (ds['time.month']==month) & (ds['time.day']==day), drop=True)
    return data


def _preprocess_coords_aggregate(x, coords, function, start_year = 2002, end_year = 2021):
    ds = x.where((x['time.year'] > start_year) & (x['time.year'] < end_year), drop=True)
    
    # ONE INDEX WITH ONLY THE NECESSARY COORDINATES
    data = ds.sel(latitude = coords['lat'].to_xarray(), longitude = coords['lon'].to_xarray(), method = 'nearest')
    
    var = list(data.keys())[0]
    data = function(data, var)

    return data

def _preprocess_bounds_aggregate(x, lon_bnds, lat_bnds, function, start_year = 2002, end_year = 2021):
    ds = x.where((x['time.year'] > start_year) & (x['time.year'] < end_year), drop=True)

    # Remove unwanted data by coordinates
    data = ds.where((ds.latitude<=lat_bnds[1]) & (ds.latitude>=lat_bnds[0]) & (ds.longitude<=lon_bnds[1]) & (ds.longitude>=lon_bnds[0]), drop=True)

    var = list(data.keys())[0]
    data = function(data, var)

    return data


def get_means(data, var, time='1m'):
    y = data[var]
    data_arr = y.resample(time=time).mean()
    data = data_arr.to_dataset()
    return data

def get_sums(data, var, time='1m'):
    y = data[var]
    data_arr = y.resample(time=time).sum()
    data = data_arr.to_dataset()
    return data

def get_frost_days(data, var, time='1m'):
    data = data.assign(frost_days = lambda x: x[var] < 0)
    return get_sums(data, 'frost_days', time)

def get_par(data, var, time='1m'):
    # CONVERT DAILY qq TO rss 
    data = data.assign(rss = lambda x: x[var] *0.0864)
    # CONVERT DAILY rss TO MONTHLY MEANS
    data = get_means(data, 'rss', time)
    # CONVERT rss TO par
    data = data.assign(par = lambda x: x['rss'] *0.44*4.56)
    return data

def get_vpd(df, tair, rh):
    tair = np.array(df[tair])
    rh = np.array(df[rh])
    applyvpd = np.vectorize(calculate_vpd_from_np_arrays)
    vpd = applyvpd(tair,rh)
    df = df.assign(vpd=vpd)
    return df

def calculate_vpd_from_np_arrays(tair,rh):
    svp = 610.7 * 10**(7.5*tair/(237.3+tair))
    vpd = svp * (1-(rh/100)) / 1000
    return vpd


# GET ONLY NECESSARY COORDS
def process_vars2(vars, coords):
# PROCESS EACH VAR
    for v in vars:
        print(f'Processing variable {v[0]}...')
        var = v[0]
        function = v[1]
        path = f"data/netcdf/vars/{var}/"


        partial_func = partial(_preprocess_coords_aggregate, coords=coords, function=function)

        # OPEN ALL DATASETS AT ONCE
        data = xr.open_mfdataset(
            f"{path}*.nc", combine='nested', concat_dim='time', preprocess=partial_func, chunks='auto'
        )

        netcdfs.append(data)
        print(f'{v[0]} done.')
        
    return netcdfs

# COMBINE ALL VARS INTO ONE NETCDF WITH COORDINATES WITHIN BOUNDS
def process_vars_and_aggregate(vars, lat_bnds, lon_bnds, var_path):
# PROCESS EACH VAR
    for v in vars:
        print(f'Processing variable {v[0]}...')
        var = v[0]
        function = v[1]
        path = f"{var_path}/{var}/"

        partial_func = partial(_preprocess_bounds_aggregate, lat_bnds=lat_bnds, lon_bnds=lon_bnds, function=function)

        # OPEN ALL DATASETS AT ONCE
        data = xr.open_mfdataset(
            f"{path}*.nc", combine='nested', concat_dim='time', preprocess=partial_func, chunks='auto'
        )

        data = uf.round_coords(data, lat = 'latitude', lon = 'longitude')

        netcdfs.append(data)
        print(f'{v[0]} done.')

    return netcdfs