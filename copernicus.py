import xarray as xr
import pandas as pd
import numpy as np
from functools import partial
from math import radians, cos, sin, asin, sqrt
import os



# HAVERSINE DISTANCE
# CODE FROM: 
# https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546
def dist(lat1, long1, lat2, long2):
    """
    Replicating the same formula as mentioned in Wiki
    """
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

# FIND NEAREST COORDINATES TO GIVEN COORDINATES IN A DATAFRAME 
# CODE FROM: 
# https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546
def find_nearest_coords(df, lat, long):
    distances = df.apply(
        lambda row: dist(lat, long, row['lat'], row['lon']), 
        axis=1)
    row = df.loc[distances.idxmin()]
    return row['climID']


def _preprocess1(x, lon_bnds, lat_bnds, year, month=1, day=1):

    # Remove unwanted data by coordinates
    ds = x.where((x.latitude<=lat_bnds[1]) & (x.latitude>=lat_bnds[0]) & (x.longitude<=lon_bnds[1]) & (x.longitude>=lon_bnds[0]), drop=True)
    
    # 1 day of data to get all coordinates
    data = ds.where((ds['time.year']==year) & (ds['time.month']==month) & (ds['time.day']==day), drop=True)
    return data


def _preprocess2(x, coords, function):
    ds = x.where((x['time.year']>2002) & (x['time.year']<2021), drop=True)
    
    # ONE INDEX WITH ONLY THE NECESSARY COORDINATES
    data = ds.sel(latitude = coords['lat'].to_xarray(), longitude = coords['lon'].to_xarray(), method = 'nearest')
    
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

# CONVERT TO DF
def convert_copernicus_to_df(dataset):
    df = dataset.to_dataframe()
    new = df.reset_index(level=('index'))
    new.rename(columns={'latitude':'lat', 'longitude':'lon', 'index': 'climID'}, inplace=True)
    return new

# GET CLIMATE IDS FOR SITES
def climateIDs_for_sites(ref_path, sites_path):
    ref_df = pd.read_csv(ref_path, parse_dates=['time'])
    sites_df = pd.read_csv(sites_path,index_col = [0])
    sites_df['climID'] = sites_df.apply(lambda x: find_nearest_coords(ref_df, x['lat'], x['lon']), axis=1)
    # sites_df.to_csv('data/csv/sites_climid.csv')
    print(sites_df)


def get_climID_reference(path):
    return


def get_bounds(sites_path):
    return

# netcdfs = []
# # vars = [['hu', get_means], ['tg', get_means], ['tx', get_frost_days], ['qq', get_par], ['rr', get_sums]]
# vars = [['tn', get_means], ['tx', get_means], ['tg', get_means]]


# # GET RELEVANT COORDS LIST (FILTERED BY SITE FILE CLIMATE IDS)
# ref_path = f"data/csv/climateIdReference.csv"
# ref_df = pd.read_csv(ref_path, parse_dates=['time'])
# sites_path = f"data/csv/coords_climid.csv"
# sites_df = pd.read_csv(sites_path, index_col = [0])
# ids = sites_df['climID']
# coords_df = ref_df.loc[ref_df['climID'].isin(ids)]
# coords = coords_df[['lat', 'lon']]

# for v in vars:
#     print(f'Processing variable {v[0]}...')
#     var = v[0]
#     function = v[1]
#     path = f"data/copernicus_netcdf/vars/{var}/"


#     partial_func = partial(_preprocess2, coords=coords, function=function)

#     # OPEN ALL DATASETS AT ONCE
#     data = xr.open_mfdataset(
#         f"{path}*.nc", combine='nested', concat_dim='time', preprocess=partial_func, chunks='auto'
#     )

#     netcdfs.append(data)
#     print(f'{v[0]} done.')

# # GET FILES
# new = xr.merge(netcdfs, compat='override')
# print(new)

# print(f'Writing netcdf...')
# new = new.to_netcdf()
# data = xr.open_dataset(new)
# print(data)
# print(f'Done.')

# print(f'Converting to dataframe...')
# df = convert_copernicus_to_df(data)
# df['lat'] = np.around(df['lat'],decimals=2)
# df['lon'] = np.around(df['lon'],decimals=2)
# print(f'Done.')

csv_path = f'data/csv/sweden_may23_monthly_min_max_temp.csv'
prebas_path = f'data/csv/prebas_sweden_may23_monthly_weather.csv'
temp = pd.read_csv(csv_path)
df = pd.read_csv(prebas_path)
tn = temp['tn']
tx = temp['tx']
print(tn)
print(tx)
df = df.assign(t_min=tn, t_max=tx)
print(df)

# df = get_vpd(df, 'tg', 'hu')
# df = df.rename(columns={'tg' : 'tair', 'rr':'precip'})
# df = df.drop(columns=['hu', 'rss'])

# # YEAR AND MONTH TO COLS
# df = df.reset_index(level=('time'))
# year = df['time'].dt.year
# month = df['time'].dt.month
# df = df.assign(year=year, month=month)
# df = df.drop(columns=['time'])
# # REARRANGE DATAFRAME
# cols = ['year', 'month', 'climID', 'lat', 'lon', 'frost_days', 'tair', 'precip', 'par', 'vpd']
# df = df.loc[:, cols]

print(f'Writing to csv...')
csv_path = f'data/csv/prebas_sweden_may23_monthly_weather.csv'
df.to_csv(csv_path, index=False)
print(df)
print(f'Done.')


