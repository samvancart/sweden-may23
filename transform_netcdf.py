import xarray as xr
import pandas as pd
import numpy as np
from functools import partial
from math import radians, cos, sin, asin, sqrt
import os
import transform_csv as tr



def _preprocess(x, lon_bnds, lat_bnds):
    # Configure variable names
    # coords = list(x.coords)
    # ds = x
    # if 'lon' in coords or 'lat' in coords:
    #     ds = ds.rename_vars({'lon':'longitude', 'lat':'latitude'})

    # Remove unwanted data by coordinates
    ds = x.where((x.latitude<=lat_bnds[1]) & (x.latitude>=lat_bnds[0]) & (x.longitude<=lon_bnds[1]) & (x.longitude>=lon_bnds[0]), drop=True)
    # ds = x.where((x.lat<=lat_bnds[1]) & (x.lat>=lat_bnds[0]) & (x.lon<=lon_bnds[1]) & (x.lon>=lon_bnds[0]), drop=True)

    # Remove unwanted years
    # data = ds.where((ds['time.year']>=2022) & (ds['time.month']==1) & (ds['time.day']==1),drop=True)
    data = ds.where((ds['time.year']>=2022) & (ds['time.month']==1),drop=True)
    # data['frost'] = data.where(data['tx'] < 0, data['frost'] == 1, data['frost'] == 0)
    # y = x['RR']
    # Return daily sums
    # sums = y.resample(time="1d").sum()
    # Return daily means
    # means = x[['T2M', 'GL', 'RH2M']]
    # means = means.resample(time="1d").mean()

    # return xr.merge([means,sums])
    return data
    # return ds


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



# CONVERT TO DF
def convert_copernicus_to_df(dataset, vars):
    df = dataset.to_dataframe()
    df.reset_index(inplace=True)
    df.set_index(df['time'], inplace=True)
    df=df[vars]
    df.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
    return df

# TRANSFORM DATAFRAME
def transform_df(df, columns, rename_cols):
    df.reset_index(inplace=True)
    df.set_index(df['time'], inplace=True)
    df=df[columns]
    df = df.rename(columns=rename_cols)
    return df

# WRITE TO CSV
# csv_file = 'data/csv/INCAL_DAILY_T2M_2015_styria_grouped.csv'
def write_to_csv(df, path):
    df.to_csv(path)

def mask_by_coords(df,lat_bounds,lon_bounds):
    lat_mask = (df['lat'] >= lat_bounds[0]) & (df['lat']<=lat_bounds[1])
    lon_mask = (df['lon'] >= lon_bounds[0]) & (df['lon']<=lon_bounds[1])
    coord_mask = lat_mask & lon_mask

    return coord_mask


def nearest_x(df,x,column):
    nearest_lat  = x
    arr          = np.array(df[column])
    diff_arr     = np.absolute(arr-nearest_lat)
    index        = np.argmin(diff_arr)
    coord        = arr[index]
    return coord

def get_files_in_folder(path):
    files = []
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            files.append(f)
    return files

years = "2011-2022"
file = "tx_ens_mean_0.1deg_reg_2011-2022_v27.0e.nc"
path = f"data/copernicus_netcdf/{years}/{file}"
path = f"data/copernicus_netcdf/{years}/"

# BOX BOUNDARIES
# SWEDEN BOUNDS
lon_bnds, lat_bnds = (10.9, 24.2), (54.9, 69)
partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

# OPEN ALL DATASETS AT ONCE
# data = xr.open_mfdataset(
#     f"{path}*.nc", combine='by_coords', preprocess=partial_func
#     # path, combine='nested', concat_dim='time', preprocess=partial_func
#     # f"{path}*.nc", combine='nested', concat_dim='time'
# )

# OPEN ONE DATASET 
# data = xr.open_dataset(path)
# data = partial_func(data)

# GET FROST DAYS
# data = data.assign(frost_days = lambda x: x['tx'] < 0)
# y = data['frost_days']
# sums = y.resample(time="1m").sum()
# print(data)
# print(sums)

# vars = ['hu', 'qq' , 'rr', 'tg', 'tx', 'latitude', 'longitude']
# vars = ['frost_days', 'latitude', 'longitude']
# df = convert_copernicus_to_df(sums, vars)
# df['lat'] = np.around(df['lat'],decimals=2)
# df['lon'] = np.around(df['lon'],decimals=2)
# CLIMATE IDS
# grouped = df.groupby(["lat", "lon"],as_index=True)
# df['climID'] = grouped.grouper.group_info[0]
# print(df)
# # WRITE CSV
# csv_path = f'data/csv/test/frost_days.csv'
# df.to_csv(csv_path)


# GET CLIMATE IDS FOR SITES
ref_path = f"data/csv/climateIdReference.csv"
ref_df = pd.read_csv(ref_path, parse_dates=['time'])
sites_path = f"data/csv/coords.csv"
sites_df = pd.read_csv(sites_path,index_col = [0])
# sites_df = sites_df.head(300)
sites_df['climID'] = sites_df.apply(lambda x: find_nearest_coords(ref_df, x['lat'], x['lon']), axis=1)
sites_df.to_csv('data/csv/coords_climid.csv')
print(sites_df)

# GET RELEVANT COORDS LIST (FILTERED BY SITE FILE CLIMATE IDS)

# INCA ONE YEAR NETCDF TO PREBAS CSV
def inca_netcdf_to_prebas():

    for i, y in enumerate(range(2016,2022)):
        print(f'Processing {y}')

        # netcdf_from_file = str(y)
        # folder = 'processed'
        # path = f"data/zamg_netcdf/{folder}/{netcdf_from_file}.nc"


        # # OPEN ONE DATASET 
        # data = xr.open_dataset(path)
        # data = data.reset_index(['x','y'],drop=True)
        # df = data.to_dataframe()
        # df = df.droplevel(['x','y'])
        # grouped = df.groupby(["lat", "lon"],as_index=True)
        # df['climID'] = grouped.grouper.group_info[0]
        # df.reset_index(level=[0],inplace=True)

        # TO PREBAS

        # df['DOY'] = df['time'].dt.day_of_year

        # def calculate_vpd_from_np_arrays(tair,rh):
        #     svp = 610.7 * 10**(7.5*tair/(237.3+tair))
        #     vpd = svp * (1-(rh/100)) / 1000
        #     return vpd

        # gl = np.array(df['GL'])
        # applyrss = np.vectorize(lambda x: x*0.0864)
        # rss = applyrss(gl)

        # applypar = np.vectorize(lambda x: x *0.44*4.56)
        # par = applypar(rss)

        # tair = np.array(df['T2M'])
        # rh = np.array(df['RH2M'])
        # applyvpd = np.vectorize(calculate_vpd_from_np_arrays)
        # vpd = applyvpd(tair,rh)


        # cols = {'T2M':'TAir','RR': 'Precip'}
        # df = df.rename(columns=cols)
        # df['PAR'] = par
        # df['VPD'] = vpd

        # df = df[['time','DOY','climID','PAR', 'TAir', 'VPD', 'Precip', 'lat', 'lon']]

        # print(df)

        # year = str(y)
        # csv_path = f'data/csv/inca/prebas_inca_styria_{year}.csv'
        # df.to_csv(csv_path, index=False)
        # print(f'{y}.csv done.')



