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

from pysolar.solar import *


#cheap_node_list = ['001e06305a6b']
cheap_node_list = ['001e063059c2', '001e06305a61', '001e06305a6c', '001e06318cd1', '001e06323a05', '001e06305a57', '001e06305a6b', '001e06318c28', '001e063239e3', '001e06323a12']
node_id = '10004098'
gps_node_id = '001e0610c2e9'
dir_out = '../figures/'
dir_data = '../data/'
dir_in_cheap = '../lightsensors/'

date_start = datetime.datetime(2020,2,27)
years = ['2019','2020'] ####
months = ['1','2','3','4','5','6','7','8','9','10','11','12']
days = np.array(range(1,31+1)).astype(str) #### np.array(range(1,31+1)).astype(str)
days = list(days)

hours = (np.array(range(0,24))).astype(str)
hours = list(hours)

bins = np.array(range(0,420+1)).astype(str)
bins = list(bins)
for i in range(len(bins)):
    bins[i] = 'Spectrum[' + bins[i] + ']'

wavelengths = np.array(range(360,780+1)).astype(str)
for i in range(len(wavelengths)):
    wavelengths[i] = wavelengths[i] + 'nm'
wavelengths = list(wavelengths)


# if data has been preprocessed before, run this directly
print("Reading Minolta Data")
fn_in = '../Minolta/'+node_id+'.csv' # resampled
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
lat_delta = 0.0001

long_median = float(df_gps[['longitude']].median()) # 
long_delta = 0.0001


# drop driving data
iwant = df_minolta.index.date < datetime.date(2020, 1, 7) # gps starts from Jan 7, 2020, no driving before that
iwant += (df_minolta['latitude']  > (lat_median -lat_delta)  )\
        & (df_minolta['latitude']  < (lat_median +lat_delta*2))\
        & (df_minolta['longitude'] > (long_median-long_delta) )\
        & (df_minolta['longitude'] < (long_median+long_delta) )

iwant = ~iwant

# filtered
df_minolta = df_minolta[iwant]

# drop the gps filter
# df_minolta.drop(columns = df_gps.columns, inplace = True)
df_minolta.dropna(how='all', inplace=True)
df_minolta.drop_duplicates(inplace=True)
df_minolta.dropna(inplace=True)

fn_data = dir_data + 'driving_' + node_id + '.csv'
df_minolta.to_csv(fn_data)
            
