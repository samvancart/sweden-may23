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
def get_all_vars_netcdf_with_bounds(sites_path, vars, var_path):
    # GET INITIAL LAT LON BOUNDS FROM SITES FILE
    sites_df = pd.read_csv(sites_path)
    bnds = uf.get_bounds(sites_df)
    lat_bnds, lon_bnds = bnds[0], bnds[1]

    # PROCESS NETCDF
    netcdfs = vh.process_vars3(vars=vars, lat_bnds=lat_bnds, lon_bnds=lon_bnds, var_path=var_path)
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
    df = uf.year_and_month_to_cols(df)
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


# GET INITIAL LAT LON BOUNDS FROM SITES FILE
# sites_path = f"data/csv/coords.csv"
# sites_df = pd.read_csv(sites_path)
# bnds = get_bounds(sites_df)
# lat_bnds, lon_bnds = bnds[0], bnds[1]
# print(lat_bnds, lon_bnds)


# # DEFINE YEAR AND VARIABLE (CAN BE ANY) FOR REFERENCE CLIMID DF
# year = 1995
# var = 'tg'
# folder = f"data/copernicus_netcdf/vars/{var}/"
# netcdf_path = get_files_in_folder(folder)
# netcdf_path=netcdf_path[0]

# print('Writing reference netcdf...')
# netcdf_file = get_climID_reference_netcdf(netcdf_path, lat_bnds, lon_bnds, year)
# print('Done.')

# print('Writing reference df...')
# ref_df = get_climID_reference_df(netcdf_file)
# print('Done.')

# print('Writing climate ids for sites...')
# sites_df = climateIDs_for_sites_from_files(ref_df, sites_path)
# print(sites_df)
# print('Done.')





# sites_path = f"data/csv/coords.csv"
# temp_path = f'data/csv/temp'
# nan_removal_path = f'{temp_path}/nan_removal.csv'
# nans_removed_path = f'{temp_path}/nans_removed.csv'
# sites_ids_path = f'{temp_path}/sites_id.csv'
# filtered_df_path = f'{temp_path}/filtered_df.csv'
# prebas_path = f'{temp_path}/prebas.csv'

# # 1.
# df = get_nan_removal_df(sites_path)
# write_df_to_csv(df, nan_removal_path)

# # 2.
# df = remove_nan_ids(nan_removal_path)
# print(df)
# write_df_to_csv(df, nans_removed_path)

# # 3.
# ref_df = get_ref_df(nans_removed_path)
# print(ref_df)
# print('Writing climate ids for sites...')
# sites_df = climateIDs_for_sites_from_files(ref_df, sites_path)
# print('Done.')
# sites_df['climID'] = sites_df['climID'].astype(int)
# print(sites_df)
# write_df_to_csv(sites_df, sites_ids_path)

# # 4.
# df = get_filtered_df(nans_removed_path, sites_ids_path)
# print(df)
# write_df_to_csv(df, filtered_df_path)

# 5.
# df = pd.read_csv(filtered_df_path)
# df = get_vpd(df, 'tg', 'hu')
# df = df.rename(columns={'tg' : 'tair', 'rr':'precip', 'tx': 't_max', 'tn': 't_min'})
# df = df.drop(columns=['hu', 'rss'])
# cols = ['year', 'month', 'climID', 'lat', 'lon', 'frost_days', 'tair', 'precip', 'par', 'vpd', 't_min', 't_max']
# df = rearrange_df(df, cols)
# print(df)
# write_df_to_csv(df, prebas_path)








# process_vars3(vars=vars, lat_bnds=lat_bnds, lon_bnds=lon_bnds)

# new = xr.merge(netcdfs)
# print(new)
# data = write_netcdf(new)
# print(data)

# print(f'Converting to dataframe...')
# df = get_temp_df_for_nan_removal(data)
# df = df.set_index(['lat','lon'])
# df.sort_index(inplace=True)
# df = df.reset_index(level=('lat', 'lon'))
# df = year_and_month_to_cols(df)
# print(f'Done.')

# print(f'Writing to file...')
# df.to_csv(temp_path, index=False)
# print(f'Done.')




# df = pd.read_csv(f'data/csv/temp/nans_removed.csv')
# df = df.set_index(['lat','lon'])
# df.sort_index(inplace=True)
# df = df.reset_index(level=('lat', 'lon'))
# df = year_and_month_to_cols(df)
# df = remove_nans_from_df(df)
# df.to_csv(f'data/csv/temp/nans_removed.csv', index=False)
# print(df)







# ref_df = df[['lat','lon','climID']].copy().drop_duplicates()
# print(ref_df)

# print('Writing climate ids for sites...')
# sites_df = climateIDs_for_sites_from_files(ref_df, sites_path)
# print(sites_df)
# sites_df.to_csv(f'data/csv/temp/sites_id.csv')
# print('Done.')



# STEP 4: FILTER CLIMATE DATA BY SITE DATA CLIMATE IDS


# sites_ids_path = f'data/csv/temp/sites_id.csv'
# sites_df = pd.read_csv(sites_ids_path, index_col=0)
# sites_df['climID'] = sites_df['climID'].astype(int)
# ids = sites_df['climID'].unique()
# df = df[df['climID'].isin(ids)]

# print(sites_df)
# print(df)



# # GET RELEVANT COORDS LIST (FILTERED BY SITE FILE CLIMATE IDS)
# # ref_path = f"data/csv/climateIdReference.csv"
# # ref_df = pd.read_csv(ref_path, parse_dates=['time'])
# # sites_path = f"data/csv/coords_climid.csv"
# # sites_df = pd.read_csv(sites_path, index_col = [0])
# ids = sites_df['climID']
# coords_df = ref_df.loc[ref_df['climID'].isin(ids)]
# coords = coords_df[['lat', 'lon']]






# # GET FILES
# netcdfs = process_vars2(vars, coords)
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

# STEP 5: CALCULATE VPD AND REARRANGE COLUMNS

# df = get_vpd(df, 'tg', 'hu')
# df = df.rename(columns={'tg' : 'tair', 'rr':'precip', 'tx': 't_max', 'tn': 't_min'})
# df = df.drop(columns=['hu', 'rss'])

# # YEAR AND MONTH TO COLS
# df = df.reset_index(level=('time'))
# year = df['time'].dt.year
# month = df['time'].dt.month
# df = df.assign(year=year, month=month)
# df = df.drop(columns=['time'])

# # REARRANGE DATAFRAME
# cols = ['year', 'month', 'climID', 'lat', 'lon', 'frost_days', 'tair', 'precip', 'par', 'vpd', 't_min', 't_max']
# df = df.loc[:, cols]

# print(f'Writing to csv...')
# csv_path = f'data/csv/test/prebas_sweden_may23_monthly_weather_test_buffer_0.5.csv'
# df.to_csv(csv_path, index=False)
# print(df)
# print(f'Done.')
