import pandas as pd
import numpy as np
import pyproj


def coords_from_lambert(row):
    x = row['X_LAMBERT_R']*1000
    y = row['Y_LAMBERT_R']*1000
    transformer = pyproj.Transformer.from_crs("epsg:3416", "epsg:4326")
    return transformer.transform(x, y)

def get_bounds(lat, lon, box_width=0.1):
    l_lat = lat-(box_width/2)
    u_lat = lat+(box_width/2)
    l_lon = lon-(box_width/2)
    u_lon = lon+(box_width/2)
    lat_bounds = np.around(l_lat, decimals=2), np.around(u_lat,decimals=2)
    lon_bounds = np.around(l_lon,decimals=2), np.around(u_lon,decimals=2)
    return lat_bounds, lon_bounds
     
def assign_climIDs(df_cop,row,sites):
    bounds = get_bounds(df_cop['lat'][row],df_cop['lon'][row])
    lat_bnds = bounds[0]
    lon_bnds = bounds[1]
    mask = sites.where((sites['lat']<lat_bnds[1]) & (sites['lat']>=lat_bnds[0]) & (sites['lon']<lon_bnds[1]) & (sites['lon']>=lon_bnds[0]))
    mask.dropna(inplace=True)
    mask['climID'] = df_cop['climID'][row]
    return mask

# DATA PATH
# data_path = 'data/csv'

# # GET SITES FILE
# # sites_file = 'esa_sites.csv'
# # file = f'{data_path}/{sites_file}'
# # df_sites = pd.read_csv(file)
# # print(df_sites)

# # GET WEATHER FILE
# weather_file = 'copernicus_tair_styria_2011-2021.csv'
# file = f'{data_path}/{weather_file}'
# df_cop = pd.read_csv(file)
# # print(df_cop)


# # ASSIGN REFERENCE CLIMATE ID:S
# grouped = df_cop.groupby(["lat", "lon"],as_index=True)
# df_cop['climID'] = grouped.grouper.group_info[0]
# # df_cop['climID'] = df_cop['climID']+1
# # print(df_cop)


# # MOCK DF
# d = {'lat': [46.76435998,46.79171066],
#       'lon': [15.87205907,15.90352198]}
# df_sites = pd.DataFrame(data=d)

# # LENGTH OF UNIQUE CLIMATE ID:S IN REFERENCE
# ids = len(pd.unique(df_cop['climID']))

# # CREATE DATAFRAME FOR ID:S
# d = {'lat':[],'lon':[],'climID':[]}
# df_clim = pd.DataFrame(data=d)
# print(df_cop.head(ids))

# # GET CLIMATE IDS FOR SITES
# for i in range(ids):
#     data = assign_climIDs(df_cop,i,df_sites)
#     if not data.empty:
#         df_clim = pd.concat([df_clim,data])
# df_clim['climID'] = df_clim['climID'].astype(int)

# print(df_clim)

# # GET SITES CSV WITH CLIMIDS
# # sites_file = 'sites_climid.csv'
# # file = f'{data_path}/{sites_file}'
# # df_clim = pd.read_csv(file)
# # df_clim.drop(df_clim.columns[0],axis=1,inplace=True)
# # print(df_clim)
# # id = 0
# # climID = df_cop[df_cop.climID == id].iloc[0]
# # print(climID)

# # WRITE TO CSV
# # df_cop.to_csv(f"{data_path}/copernicus_tair_styria_2011-2021_climid.csv.csv")
# # df_clim.to_csv(f"{data_path}/sites_climid.csv")



