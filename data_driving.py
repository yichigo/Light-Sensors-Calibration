import pandas as pd
import numpy as np
import os
import sys
import datetime
import time
#from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

import time


minolta_node_id = '10004098'
gps_node_id = '001e0610c2e9'
dir_minolta = '../Minolta/'
dir_gps = dir_minolta

dir_data = '../data/'
dir_out = '../figures/'

# read minolta data
print("Reading Minolta Data")
fn_in = dir_minolta + minolta_node_id + '_sunPosition.csv' # resampled
df_minolta = pd.read_csv(fn_in, parse_dates=True, index_col = 'UTC')
len(df_minolta)

# read gps data
print("Reading GPS Data")
fn_in = dir_gps + gps_node_id + '.csv' # resampled
df_gps = pd.read_csv(fn_in, parse_dates=True, index_col = 'UTC')
len(df_gps)


# merge
df = pd.concat([df_minolta, df_gps], axis=1)


# driving data index
lat_median = float(df_gps[['latitude']].median()) # 32.992192
lat_delta = 0.001

long_median = float(df_gps[['longitude']].median()) # 
long_delta = 0.001

iwant = (df['latitude']  > (lat_median -lat_delta)  )\
        & (df['latitude']  < (lat_median + lat_delta))\
        & (df['longitude'] > (long_median - long_delta) )\
        & (df['longitude'] < (long_median + long_delta) )

iwant += (df.index.date == datetime.date(2020, 3, 25)) # gps drift when the sensor was in the lab

iwant = ~iwant

# filtered by gps
df = df[iwant]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# reset index, use UTC
df.reset_index(inplace = True)


# set time zone
time_start = time.time()
df['UTCtemp'] = df['UTC'].astype(str).apply(
                    lambda x:Time(x, format='iso', scale='utc')
                )
print(time.time() - time_start)

# calculate sun position, need about 20 mins
time_start = time.time()
df['sun'] = df[['UTCtemp','longitude','latitude']].apply(
                lambda x:
                coord.get_sun(x[0]).transform_to(
                    coord.AltAz(
                        location=coord.EarthLocation(lon=x[1] * u.deg, lat=x[2] * u.deg),
                        obstime=x[0]
                    )
                ), axis = 1
            )
print(time.time() - time_start)

# sun position quantities
time_start = time.time()
df['Zenith'] = df['sun'].apply(lambda x: x.zen.degree)
df['Azimuth'] = df['sun'].apply(lambda x: x.az.degree)
df['Sun Distance'] = df['sun'].apply(lambda x: x.distance.meter)
print(time.time() - time_start)


# wavelengths column names
bins = np.array(range(0,420+1)).astype(str)
bins = list(bins)
for i in range(len(bins)):
    bins[i] = 'Spectrum[' + bins[i] + ']'

wavelengths = np.array(range(360,780+1)).astype(str)
for i in range(len(wavelengths)):
    wavelengths[i] = wavelengths[i] + 'nm'
wavelengths = list(wavelengths)

columns = ['UTC','Illuminance'] + wavelengths\
        + ['latitude', 'longitude', 'altitude']\
        + ['Zenith','Azimuth','Sun Distance']

# write data
fn_data = dir_data + 'driving_' + minolta_node_id + '.csv'
df[columns].to_csv(fn_data, index = False)
            
