# HANDLE REMOVAL OF NAN VALUES


import xarray as xr
import pandas as pd

import climate_id as clid
import variable_handler as vh
import utility_functions as uf



def get_temp_df_for_nan_removal(data):
    df = data.to_dataframe()

    # ASSIGN IDS
    df = clid.assign_climIDs(df)

    # RENAME
    new = df.reset_index(level=('time', 'latitude', 'longitude'))
    new.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)

    return new



def remove_nans_from_df(df):
    # GET ALL NAN ROWS
    nans = df[df.isna().any(axis=1)]
    # LIST UNIQUE CLIMIDS THAT HAVE NAN ROWS
    nan_ids=nans['climID'].unique()
    # FILTER OUT FROM DF CLIMIDS CONTAINING NANS
    df = df[df['climID'].isin(nan_ids) == False]

    return df


# STEP 1: PREPARE DF ACCORDING TO BOUNDS

# 1A: GET NETCDF WITH ALL VARS
def get_all_vars_netcdf_with_bounds(function, vars, var_path, bnds):
    lat_bnds, lon_bnds = bnds[0], bnds[1]

    # PROCESS NETCDF
    netcdfs = function(vars=vars, lat_bnds=lat_bnds, lon_bnds=lon_bnds, var_path=var_path)
    new = xr.merge(netcdfs)
    data = uf.write_netcdf(new)

    return data
  

# 1B: CONVERT NETCDF TO DF
def get_nan_removal_df(data):
    print(f'Converting to dataframe...')
    df = get_temp_df_for_nan_removal(data)
    df = df.set_index(['lat','lon'])
    df.sort_index(inplace=True)
    df = df.reset_index(level=('lat', 'lon'))
    print(f'Done.')

    return df


# STEP 2: REMOVE ALL CLIMIDS WITH NANS 

def remove_nan_ids(nan_removal_path):
    df = pd.read_csv(nan_removal_path)
    df = remove_nans_from_df(df)
    return df

# STEP 3: MAKE REFERENCE DF AND ASSIGN CLIMIDS FOR SITES

def get_ref_df(path, lat = 'lat', lon = 'lon', id = 'climID'):
    df = pd.read_csv(path)
    ref_df = df[[lat, lon, id]].copy().drop_duplicates()
    return ref_df



# STEP 4: FILTER CLIMATE DATA BY SITE DATA CLIMATE IDS

def get_filtered_df(nans_removed_path, sites_ids_path, id = 'climID'):
    sites_df = pd.read_csv(sites_ids_path, index_col=0)
    df = pd.read_csv(nans_removed_path)

    sites_df[id] = sites_df[id].astype(int)
    ids = sites_df[id].unique()
    df = df[df[id].isin(ids)]

    return df



