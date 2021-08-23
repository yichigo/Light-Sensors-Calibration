import pandas as pd
import numpy as np

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

import os
from datetime import datetime
import time

node_id = '10004098'
dir_in = '../Minolta/'
dir_out = dir_in
df = pd.read_csv(dir_in+node_id+'.csv')


# Add Zenith Angle for the following fixed location
latitude = 32+59.53/60  # or use median 32.992192
longitude = -(96+45.47/60) # or use median -96.757845

# set time zone
print('Formating UTC string ...')
time_start = time.time()
df['UTCtemp'] = df['UTC'].astype(str).apply(lambda x:Time(x, format='iso', scale='utc'))
print(time.time() - time_start)


# calculate sun position, need 1-2 hours
print('Calculating Solar Position ...')
time_start = time.time()
location = coord.EarthLocation(lon=longitude * u.deg, lat=latitude * u.deg)
df['sun'] = df['UTCtemp'].apply(lambda x:\
                                coord.get_sun(x).transform_to(coord.AltAz(location=location,obstime=x))
                               )
print(time.time() - time_start)

# sun position quantities
print('Calculating Zenith angle and Azimuth angle ...')
time_start = time.time()
df['Zenith'] = df['sun'].apply(lambda x: x.zen.degree)
df['Azimuth'] = df['sun'].apply(lambda x: x.az.degree)
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


# write data
columns = ['UTC','Illuminance'] + wavelengths + ['Zenith','Azimuth']
df[columns].to_csv(dir_out+node_id+'_solarAngle.csv', index = False)
