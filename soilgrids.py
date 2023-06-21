# get soil data from soilgrids api


import numpy as np
import pandas as pd
import requests

'lon=17.0135467086715&lat=63.9668856940176&property=cfvo&depth=0-5cm&depth=0-30cm&depth=5-15cm&depth=15-30cm&depth=30-60cm&depth=60-100cm&depth=100-200cm&value=mean'


# DEFAULT PARAMETERS

def_url = 'https://rest.isric.org/soilgrids/v2.0/properties/query?'
def_lon = 17.0135467086715
def_lat = 63.9668856940176
def_properties = [ "bdod", "cec", "cfvo", "clay", "nitrogen", "ocd", "ocs", "phh2o", "sand", "silt", "soc" ]
def_depths =  [ "0-5cm", "0-30cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm" ]
def_values =  [ "Q0.05", "Q0.5", "Q0.95", "mean", "uncertainty" ]


def build_api_call(
    base_url = def_url,
    lon = def_lon,
    lat = def_lat,
    properties = def_properties,
    depths = def_depths,
    values = def_values

):
    str_lon = f'lon={str(lon)}'
    str_lat = f'lat={str(lat)}'
    str_properties = list(map(lambda x: 'property='+ x, properties))
    str_depths = list(map(lambda x: 'depth='+ x, depths))
    str_values = list(map(lambda x: 'value='+ x, values))

    joined = [str_lon] + [str_lat] + str_properties + str_depths + str_values
    req_str = '&'.join(joined)
    api_call = base_url + req_str

    return api_call


def req_data(url):
    """ Requests data from api

    Parameters:
        url (string): request url

        Returns:
            response (Requests.models.response object): response from api

    """
    try:
        r = requests.get(url)
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    return r

# GET DATA AS RESPONSE CONTENT
def get_data(req):
    """ Gets content from response object

        Parameters:
            req: The request url.

        Returns:
            Response content: Response content as dict. None if no content in response. 
    
    """

    # print(f'Fetching data from {req}...')
    res = req_data(req)
    if res.status_code == 200:
        content = res.json()
        # print("Fetched data succesfully.")
        return content
    else:
        print(f'An error occured with status code {res.status_code}.')
        print(res.content)
        return None
    
data = []

def get_data_list(lon, lat):
    request = build_api_call(lon=lon , lat=lat,properties=['cfvo'], values=['mean'])
    res = get_data(request)

    coords = res.get('geometry').get('coordinates')
    depths = res.get('properties').get('layers')[0].get('depths')

    data_row = coords

    for item in depths:
        data_row.append(item['values'].get('mean'))

    return data_row

# sites = pd.read_csv('data/csv/temp/sites_id.csv')
# # sites = sites.head(50)
# colnames = ['lon', 'lat', '0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']

# lon = np.array(sites['lon'])
# lat = np.array(sites['lat'])

# print(f'Fetching data...')
# lonlats = tuple(zip(lon,lat))
# for i, item in enumerate(lonlats):
#     data.append(get_data_list(item[0], item[1]))
#     print(i)

# print(f'Done.')
# df = pd.DataFrame(data=data, columns=colnames)
# df.to_csv('data/csv/cfvo_soildata.csv',index = False)
# print(df)
