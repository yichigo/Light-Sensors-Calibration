import pandas as pd
import numpy as np
import pytz
import os

# GPS data
node_id = '001e0610c2e9'
dir_in = '/Volumes/Backup Plus/MINTS/Minolta/'+ node_id + '/'
dir_out = '../Minolta/'
sensor_name = 'GPGGA'

days = []
for i in range(1,31+1):
    day = str(i)
    if len(day)<2:
        day = '0'+day
    days.append(day)

years = ['2020'] #### no data in 2019
months = ['1','2','3','4','5','6','7','8','9','10','11','12']


df_gps = pd.DataFrame()
for year in years:
    for month in months:
        if len(month) == 1:
            month = '0' + month
        for day in days:
            if len(day) == 1:
                day = '0' + day
            dirname = dir_in+year+'/'+month+'/'+day+'/'
            if not os.path.isdir(dirname):
                continue
            print(dirname)
            
            filename = dirname+'MINTS_'+node_id+'_'+sensor_name+'_'+year+'_'+month+'_'+day+'.csv'
            if not os.path.isfile(filename):
                continue
            print(filename)
            df1 = pd.read_csv(filename, usecols = ['dateTime','latitude', 'longitude', 'altitude'], parse_dates=True, index_col = 'dateTime')
            df1.index.name = 'UTC'
            # merge into df
            if len(df_gps)==0:
                df_gps = df1
            else:
                df_gps = pd.concat([df_gps, df1])


df_gps['latitude'] = (df_gps['latitude'] // 100) + (df_gps['latitude'] % 100)/60
df_gps['longitude'] = (df_gps['longitude'] // 100) + (df_gps['longitude'] % 100)/60
df_gps['longitude'] = -df_gps['longitude']
print(df_gps.head())

print("Median Location:")
print(df_gps[['latitude']].median())
print(df_gps[['longitude']].median())


print('Length: ', len(df_gps))
df_gps_resample = df_gps.resample('10S').mean()
print('Length resample: ', len(df_gps_resample))
df_gps_resample = df_gps_resample.dropna(axis=0,how='all')
print('Length dropna: ', len(df_gps_resample))


df_gps_resample.to_csv(dir_out+node_id+'.csv')
print(df_gps_resample.head())

