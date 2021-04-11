import pandas as pd
import numpy as np

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

import os
from datetime import datetime

node_id = '10004098'
dir_in = '/Volumes/Backup Plus/MINTS/Minolta/'+node_id+'/'
dir_out = '../Minolta/'

years = ['2019', '2020'] ####
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


df = pd.DataFrame([])
for year in years:
    for month in months[:]:
        for day in days[:]:
            dirname = dir_in+year+'/'+month+'/'+day+'/'
            if not os.path.isdir(dirname):
                continue
            print(dirname)

            for hour in hours:
                filename = dirname+'MINTS_Minolta_'+node_id+'_'+year+'_'+month+'_'+day+'_'+hour+'.csv'
                if not os.path.isfile(filename):
                    continue

                # check the size of file, skip if the size > 100 mb, which is because Minolta sensor stuck
                if os.stat(filename).st_size > 100*1000*1000:
                    continue

                # read data
                df1 = pd.read_csv(filename)

                # drop duplicat. Minolta sensor may stuck and generate duplicate data
                df1.drop_duplicates(subset=[' Illuminance', ' Tcp'], keep = 'first', inplace = True) #  Correlated color temperature (Tcp)

                # merge datetime
                df1['UTC'] = pd.to_datetime(df1['Date']+' '+df1[' Time'])
                
                ##### datetime calibration start #####
                ##### in 2019, 2020, the datetime was from Minolta sensor and has error
                ##### remove this process in 2021. In the new version of 2021, datetime is from the PC connected to internet
                mtime = datetime.utcfromtimestamp(os.path.getmtime(filename))
                time_more = df1['UTC'].iloc[-1] - mtime
                
                if abs(time_more.seconds) > 3600:
                    print("ERROR of time:", filename)
                df1['UTC'] = df1['UTC'] - time_more
                ##### datetime calibration end #####

                # merge df1 into df
                if len(df)==0:
                    df = df1
                else:
                    df = pd.concat([df, df1])

#df['UTC'] = pd.to_datetime(df['Date']+' '+df[' Time'])
df = df[['UTC',' Illuminance']+bins] # there is a space in front of variable Illuminance
df = df.set_index('UTC')

df.columns = ['Illuminance'] + wavelengths
print('data length: ', len(df))

print('dropna for all NaN')
df.dropna(how = "all", inplace = True)
print('data length: ', len(df))

print('drop_duplicates')
df.drop_duplicates(inplace=True)
print('data length: ', len(df))

print(df.head())
df.to_csv(dir_out+node_id+'_raw.csv')


# df = pd.read_csv(dir_out+node_id+'_raw.csv', parse_dates=True, index_col = 'UTC')

############### Resample the data by 10 s, and add Zenith Angle ##############

df_resample = df.resample('10S').mean()
#df_resample = df.resample('30S', label = 'left', loffset = '15S' ).mean()
df_resample = df_resample.dropna(axis=0,how='all')
print(len(df_resample))
print(df_resample.head())

df_resample.to_csv(dir_out+node_id+'.csv')
