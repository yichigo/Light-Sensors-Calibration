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


#cheap_node_list = ['001e06305a6b']
cheap_node_list = ['001e063059c2', '001e06305a61', '001e06305a6c', '001e06318cd1', '001e06323a05', '001e06305a57', '001e06305a6b', '001e06318c28', '001e063239e3', '001e06323a12']
node_id = '10004098'
gps_node_id = '001e0610c2e9'
dir_out = '../figures/'
dir_data = '../data/'
dir_in_cheap = '../lightsensors/'


# if data has been preprocessed before, run this directly
print("Reading Minolta Data")
fn_in = '../Minolta/'+node_id+'_solarAngle.csv' # resampled
df_minolta = pd.read_csv(fn_in, parse_dates=True, index_col = 'UTC')


# read gps data
print("Reading GPS Data")
fn_in = '../Minolta/'+gps_node_id+'.csv' # resampled
df_gps = pd.read_csv(fn_in, parse_dates=True, index_col = 'UTC')
len(df_gps)


df_minolta = pd.concat([df_minolta, df_gps], axis=1)
df_minolta.dropna(how='all', inplace=True)
df_minolta.drop_duplicates(inplace=True)


lat_median = float(df_gps[['latitude']].median()) # 32.992192
lat_delta = 0.001

long_median = float(df_gps[['longitude']].median()) # -96.757845
long_delta = 0.001


# drop driving data
iwant = df_minolta.index.date < datetime.date(2020, 1, 7) # gps starts from Jan 7, 2020, no driving before that
iwant += (df_minolta['latitude']  > (lat_median -lat_delta)  )\
        & (df_minolta['latitude']  < (lat_median +lat_delta))\
        & (df_minolta['longitude'] > (long_median-long_delta) )\
        & (df_minolta['longitude'] < (long_median+long_delta) )

# filtered
df_minolta = df_minolta[iwant]

# drop the gps filter
df_minolta.drop(columns = df_gps.columns, inplace = True)
df_minolta.dropna(how='all', inplace=True)
df_minolta.drop_duplicates(inplace=True)

# import features from cheap sensors
print("Merge Data of Cheap Sensors")
for cheap_node_id in cheap_node_list:
    print(cheap_node_id)
    fn_in_cheap = dir_in_cheap + 'MINTS_' + cheap_node_id + '.csv'

    df_cheap = pd.read_csv(fn_in_cheap, parse_dates=True, index_col = 'UTC')
    #df_cheap.head()

    # Merge data
    df_all = pd.concat([df_cheap, df_minolta], axis=1)
    df_all.dropna(how='all', inplace=True)
    df_all.drop_duplicates(inplace=True)
    #df_all.to_csv(dir_data + node_id + '_'+ cheap_node_id +'.csv')
    # try to drop the SKYCAM and GPSGPGGA2

    #features_drop = ['Latitude','Longitude','Altitude']
    #df_all.drop(columns = features_drop, inplace = True)

    # fill null values for SKYCAM
    features_interpolate = ['cloudPecentage', 'allRed', 'allGreen', 'allBlue', 'skyRed', 'skyGreen',
	       'skyBlue', 'cloudRed', 'cloudGreen', 'cloudBlue']
    for var in features_interpolate:
        df_all[var].interpolate(method='linear', inplace = True)

    df_all.dropna(inplace = True)
    print(df_all.tail())
    print(len(df_all))

    fn_data = dir_data + node_id + '_'+ cheap_node_id +'.csv'
    df_all.to_csv(fn_data)
