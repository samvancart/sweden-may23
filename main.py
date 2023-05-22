import requests
import xarray as xr
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
import io

import transform_csv as tr
import transform_netcdf as trn


from dateutil import rrule
from datetime import datetime



def main():
    a = '20200101'
    b = '20211231'
    csv_path = "data/csv/coords.csv"
    df = pd.read_csv(csv_path)
    max_long = df["long"].max()
    min_long = df["long"].min()
    max_lat = df["lat"].max()
    min_lat = df["lat"].min()
    print(f"lat: {max_lat} {min_lat}")
    print(f"long: {max_long} {min_long}")

    print(df.loc[df["lat"]==max_lat])
    print(df.loc[df["lat"]==min_lat])
    print(df.loc[df["long"]==max_long])
    print(df.loc[df["long"]==min_long])



if __name__ == '__main__':
    main()
