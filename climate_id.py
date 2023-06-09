# ASSIGN CLIMATE IDS TO SITES


import xarray as xr
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from functools import partial

import variable_handler as vh

# HAVERSINE DISTANCE
# CODE FROM: 
# https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546
def dist(lat1, lon1, lat2, lon2):
    """
    Replicating the same formula as mentioned in Wiki
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

# FIND NEAREST COORDINATES TO GIVEN COORDINATES IN A DATAFRAME 
# CODE FROM: 
# https://medium.com/analytics-vidhya/finding-nearest-pair-of-latitude-and-longitude-match-using-python-ce50d62af546
def find_nearest_coords(df, latitude, longitude, lat = 'lat', lon = 'lon'):
    distances = df.apply(
        lambda row: dist(latitude, longitude, row[lat], row[lon]), 
        axis=1)
    row = df.loc[distances.idxmin()]
    return row['climID']


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
 
    return sites_df

# GET NETCDF WITH LAT AND LON BOUNDS FOR 1 DAY
def get_climID_reference_netcdf(netcdf_path, lat_bnds, lon_bnds, year):
    partial_func = partial(vh._preprocess_1_day_bounds, lat_bnds=lat_bnds, lon_bnds=lon_bnds, year=year)
    data = xr.open_dataset(netcdf_path)
    data = partial_func(data)
    new = data.to_netcdf()
    
    return new

def get_climID_reference_df(netcdf_file):
    data = xr.open_dataset(netcdf_file)
    df = data.to_dataframe()

    # GROUP BY LAT LON AND ASSIGN IDS
    df = assign_climIDs(df)

    # RENAME AND REMOVE VAR COLUMN
    new = df.reset_index(level=('time', 'latitude', 'longitude'))
    new.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
    new = new[['time', 'lat','lon', 'climID']]

    return new


# GROUP BY LAT LON AND ASSIGN IDS
def assign_climIDs(df, lat = 'latitude', lon = 'longitude', id = 'climID'):
    grouped = df.groupby([lat, lon], as_index=True)
    df[id] = grouped.grouper.group_info[0]

    return df