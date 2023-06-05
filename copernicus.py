# E-OBS weather data to prebas format. Default time period = 1 month.



import xarray as xr
import pandas as pd
import numpy as np
from functools import partial
from math import radians, cos, sin, asin, sqrt
import os



def get_files_in_folder(path):
    files = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            files.append(f)
    return files

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

# GET CLIMATE IDS FOR SITES FROM CSV
def climateIDs_for_sites_from_csv(ref_path, sites_path):
    ref_df = pd.read_csv(ref_path, parse_dates=['time'])
    sites_df = pd.read_csv(sites_path,index_col = [0])
    sites_df['climID'] = sites_df.apply(lambda x: find_nearest_coords(ref_df, x['lat'], x['lon']), axis=1)
    # sites_df.to_csv('data/csv/sites_climid.csv')
    print(sites_df)

# GET CLIMATE IDS FOR SITES
def climateIDs_for_sites_from_files(ref_df, sites_path):
    sites_df = pd.read_csv(sites_path,index_col = [0])
    sites_df['climID'] = sites_df.apply(lambda x: find_nearest_coords(ref_df, x['lat'], x['lon']), axis=1)
    # sites_df.to_csv('data/csv/coords_climid_nans_removed.csv')
    return sites_df

# GET NETCDF WITH LAT AND LON BOUNDS FOR 1 DAY
def get_climID_reference_netcdf(netcdf_path, lat_bnds, lon_bnds, year):
    partial_func = partial(_preprocess1, lat_bnds=lat_bnds, lon_bnds=lon_bnds, year=year)
    data = xr.open_dataset(netcdf_path)
    data = partial_func(data)
    new = data.to_netcdf()
    
    return new

def get_climID_reference_df(netcdf_file):
    data = xr.open_dataset(netcdf_file)
    df = data.to_dataframe()

    # GROUP BY LAT LON AND ASSIGN IDS
    grouped = df.groupby(["latitude", "longitude"], as_index=True)
    df['climID'] = grouped.grouper.group_info[0]

    # RENAME AND REMOVE VAR COLUMN
    new = df.reset_index(level=('time', 'latitude', 'longitude'))
    new.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
    new = new[['time', 'lat','lon', 'climID']]

    return new


def get_bounds(sites_df, buffer=0.1):
    lat = sites_df.lat
    lon = sites_df.lon
    lat_bounds = (lat.min()-buffer, lat.max()+buffer)
    lon_bounds = (lon.min()-buffer, lon.max()+buffer)
    return lat_bounds, lon_bounds


# GET INITIAL LAT LON BOUNDS FROM SITES FILE
sites_path = f"data/csv/coords.csv"
sites_df = pd.read_csv(sites_path)
bnds = get_bounds(sites_df)
lat_bnds, lon_bnds = bnds[0], bnds[1]
print(lat_bnds, lon_bnds)

# DEFINE YEAR AND VARIABLE (CAN BE ANY) FOR REFERENCE CLIMID DF
year = 1995
var = 'tg'
folder = f"data/copernicus_netcdf/vars/{var}/"
netcdf_path = get_files_in_folder(folder)
netcdf_path=netcdf_path[0]

print('Writing reference netcdf...')
netcdf_file = get_climID_reference_netcdf(netcdf_path, lat_bnds, lon_bnds, year)
print('Done.')

print('Writing reference df...')
ref_df = get_climID_reference_df(netcdf_file)
print('Done.')

print('Writing climate ids for sites...')
sites_df = climateIDs_for_sites_from_files(ref_df, sites_path)
print(sites_df)
print('Done.')

# NETCDF FOR EACH VARIABLE
netcdfs = []
# LIST OF VARIABLES. EACH ITEM INCLUDES VARIABLE NAME AND SPECIFIC FUNCTION 
vars = [['hu', get_means], ['tg', get_means], ['tx', get_frost_days], ['qq', get_par], ['rr', get_sums], ['tn', get_means], ['tx', get_means]]
# vars = [['tn', get_means], ['tx', get_means], ['tg', get_means]]


# GET RELEVANT COORDS LIST (FILTERED BY SITE FILE CLIMATE IDS)
# ref_path = f"data/csv/climateIdReference.csv"
# ref_df = pd.read_csv(ref_path, parse_dates=['time'])
# sites_path = f"data/csv/coords_climid.csv"
# sites_df = pd.read_csv(sites_path, index_col = [0])
ids = sites_df['climID']
coords_df = ref_df.loc[ref_df['climID'].isin(ids)]
coords = coords_df[['lat', 'lon']]

def process_vars2(vars, coords):
# PROCESS EACH VAR
    for v in vars:
        print(f'Processing variable {v[0]}...')
        var = v[0]
        function = v[1]
        path = f"data/copernicus_netcdf/vars/{var}/"


        partial_func = partial(_preprocess2, coords=coords, function=function)

        # OPEN ALL DATASETS AT ONCE
        data = xr.open_mfdataset(
            f"{path}*.nc", combine='nested', concat_dim='time', preprocess=partial_func, chunks='auto'
        )

        netcdfs.append(data)
        print(f'{v[0]} done.')
        return netcdfs

# GET FILES
netcdfs = process_vars2(vars, coords)
new = xr.merge(netcdfs, compat='override')
print(new)

print(f'Writing netcdf...')
new = new.to_netcdf()
data = xr.open_dataset(new)
print(data)
print(f'Done.')

print(f'Converting to dataframe...')
df = convert_copernicus_to_df(data)
df['lat'] = np.around(df['lat'],decimals=2)
df['lon'] = np.around(df['lon'],decimals=2)
print(f'Done.')

df = get_vpd(df, 'tg', 'hu')
df = df.rename(columns={'tg' : 'tair', 'rr':'precip', 'tx': 't_max', 'tn': 't_min'})
df = df.drop(columns=['hu', 'rss'])

# YEAR AND MONTH TO COLS
df = df.reset_index(level=('time'))
year = df['time'].dt.year
month = df['time'].dt.month
df = df.assign(year=year, month=month)
df = df.drop(columns=['time'])

# REARRANGE DATAFRAME
cols = ['year', 'month', 'climID', 'lat', 'lon', 'frost_days', 'tair', 'precip', 'par', 'vpd', 't_min', 't_max']
df = df.loc[:, cols]

print(f'Writing to csv...')
csv_path = f'data/csv/test/prebas_sweden_may23_monthly_weather_test_buffer_0.5.csv'
df.to_csv(csv_path, index=False)
print(df)
print(f'Done.')
