import xarray as xr
import numpy as np
import pandas as pd
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


# GETS MIN AND MAX LAT AND LON FROM SITES FILE
# ADDS BUFFER TO MAX VALS AND SUBTRACTS BUFFER FROM MIN VALS TO GET COORDINATE BOUNDS
def get_bounds(sites_df, buffer=0.1):
    lat = sites_df.lat
    lon = sites_df.lon
    lat_bounds = (lat.min()-buffer, lat.max()+buffer)
    lon_bounds = (lon.min()-buffer, lon.max()+buffer)
    return lat_bounds, lon_bounds



def round_coords(d, lat = 'lat', lon = 'lon', decimals=2):
    d[lat] = np.around(d[lat],decimals=decimals)
    d[lon] = np.around(d[lon],decimals=decimals)
    return d

def year_and_month_to_cols(df):
    df['time'] = pd.to_datetime(df['time'])
    year = df['time'].dt.year
    month = df['time'].dt.month
    df = df.assign(year=year, month=month)
    df = df.drop(columns=['time'])
    return df


def write_netcdf(netcdf):
    print(f'Writing netcdf...')
    new = netcdf.to_netcdf()
    data = xr.open_dataset(new)
    print(f'Done.')
    return data


def write_df_to_csv(df, path, index=False):
    print(f'Writing to {path}...')
    df.to_csv(path, index=index)
    print(f'Done.')


def rearrange_df(df, columns):
    return df.loc[:, columns]

# CONVERT TO DF
def convert_copernicus_to_df(dataset):
    df = dataset.to_dataframe()
    new = df.reset_index(level=('index'))
    new.rename(columns={'latitude':'lat', 'longitude':'lon', 'index': 'climID'}, inplace=True)
    return new