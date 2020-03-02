import pandas as pd
import numpy as np
from pysolar.solar import *
import pytz
import os

node_id = '10004098'
dir_in = '../Minolta/'+node_id+'/'
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


df = []
for year in years:
    for month in months:
        for day in days:
            dirname = dir_in+year+'/'+month+'/'+day+'/'
            if not os.path.isdir(dirname):
                continue
            print(dirname)

            for hour in hours:
                filename = dirname+'MINTS_Minolta_'+node_id+'_'+year+'_'+month+'_'+day+'_'+hour+'.csv'
                if not os.path.isfile(filename):
                    continue
                df1 = pd.read_csv(filename)
                # merge into df
                if len(df)==0:
                    df = df1
                else:
                    df = pd.concat([df, df1])

df['UTC'] = pd.to_datetime(df['Date']+' '+df[' Time'])
df = df[['UTC',' Illuminance']+bins]# there is a space in front of variable Illuminance
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



############### Resample the data by 10 s, and add Zenith Angle ##############

df_resample = df.resample('10S').mean()
#df_resample = df.resample('30S', label = 'left', loffset = '15S' ).mean()
df_resample = df_resample.dropna(axis=0,how='all')
print(len(df_resample))
print(df_resample.head())


# Add Zenith Angle
timezone = pytz.timezone("UTC")
#timezone = pytz.timezone('America/Chicago')
lat = 32+59.53/60
long = -(96+45.47/60)

zeniths = []
for i, row in df_resample.iterrows():
    date = datetime.datetime.strptime(str(i),'%Y-%m-%d %H:%M:%S')
    date = timezone.localize(date)
    zeniths.append(90.0-get_altitude(lat, long, date))
df_resample['Zenith'] = zeniths

df_resample.to_csv(dir_out+node_id+'.csv')
