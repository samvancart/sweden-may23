# E-OBS weather data to prebas format. Default time period = 1 month.
# Removes NaNs first and then creates dataframe with nearest climate ids
# INSTRUCTIONS
# Files for all the years of each e-obs variable should be in a folder with the variable's name eg. variable 'tg' in a folder called tg.
# Each step can be written as csv into a temp folder and read for the next step.
# CHECK NETCDF DIMENSIONS AFTER FIRST STEP, MIGHT BE TOO LARGE TO PROCESS FURTHER WITHOUT MODIFYING!

import xarray as xr
import pandas as pd
import numpy as np

import variable_handler as vh
import nan_handler as nh
import climate_id as clid
import utility_functions as uf



def main():
    # DEFINE NECESSARY FILE PATHS
    var_path = f'data/netcdf/vars'
    sites_path = f"data/csv/coords.csv"
    temp_path = f'data/csv/temp'
    nan_removal_path = f'{temp_path}/nan_removal.csv'
    nans_removed_path = f'{temp_path}/nans_removed.csv'
    sites_ids_path = f'{temp_path}/sites_id.csv'
    filtered_df_path = f'{temp_path}/filtered_df.csv'
    prebas_path = f'{temp_path}/prebas.csv'

    # DEFINE VARIABLE NAMES
    rh = 'hu'
    tair = 'tg'
    t_max = 'tx'
    t_min = 'tn'
    qq = 'qq'
    precip = 'rr'
    rss = 'rss'

    # LIST OF VARIABLES. EACH ITEM INCLUDES VARIABLE NAME AND AGGREGATE FUNCTION 
    vars = [[rh, vh.get_means], [tair, vh.get_means], [t_max, vh.get_frost_days], 
            [qq, vh.get_par], [precip, vh.get_sums], [t_min, vh.get_means], [t_max, vh.get_means]]

    # STEP 1: PREPARE DF ACCORDING TO BOUNDS
    sites_df = pd.read_csv(sites_path)
    bnds = uf.get_bounds(sites_df)
    data = nh.get_all_vars_netcdf_with_bounds(vh.process_vars_and_aggregate, vars, var_path, bnds)
    df = nh.get_nan_removal_df(data)
    uf.write_df_to_csv(df, nan_removal_path)

    # STEP 2: REMOVE ALL CLIMIDS WITH NANS 
    df = nh.remove_nan_ids(nan_removal_path)
    print(df)
    uf.write_df_to_csv(df, nans_removed_path)

    # STEP 3: MAKE REFERENCE DF AND ASSIGN CLIMIDS FOR SITES
    ref_df = nh.get_ref_df(nans_removed_path)
    print('ref_df')
    print(ref_df)
    print('Writing climate ids for sites...')
    sites_df = clid.climateIDs_for_sites_from_files(ref_df, sites_path)
    print('Done.')
    sites_df['climID'] = sites_df['climID'].astype(int)
    print(sites_df)
    uf.write_df_to_csv(sites_df, sites_ids_path)

    # STEP 4: FILTER CLIMATE DATA BY SITE DATA CLIMATE IDS
    df = nh.get_filtered_df(nans_removed_path, sites_ids_path)
    print(df)
    uf.write_df_to_csv(df, filtered_df_path)

    # STEP 5: CALCULATE VPD AND REARRANGE COLUMNS
    df = pd.read_csv(filtered_df_path)
    df = vh.get_vpd(df, tair, rh)
    df = df.rename(columns={tair : 'tair', precip:'precip', t_max: 't_max', t_min: 't_min'})
    df = df.drop(columns=[rh, rss])
    df = uf.year_and_month_to_cols(df)
    cols = ['year', 'month', 'climID', 'lat', 'lon', 'frost_days', 'tair', 'precip', 'par', 'vpd', 't_min', 't_max']
    df = uf.rearrange_df(df, cols)
    print(df)
    uf.write_df_to_csv(df, prebas_path)

    




if __name__ == '__main__':
    main()
