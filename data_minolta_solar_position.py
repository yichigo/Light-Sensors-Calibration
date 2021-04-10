import pandas as pd
import numpy as np

import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

import os
from datetime import datetime

node_id = '10004098'
dir_in = '../Minolta/'
dir_out = dir_in

df = pd.read_csv(dir_in+node_id+'.csv', parse_dates=True, index_col = 'UTC')


# Add Zenith Angle for the following fixed location
latitude = 32+59.53/60
longitude = -(96+45.47/60)

df_resample = df_resample.reset_index()
df_resample['UTC'] = df_resample['UTC'].astype(str)
df_resample['Zenith'] = df_resample.reset_index()['UTC'].apply(lambda x:\
                                                               coord.get_sun(Time(x, format='iso', scale='utc'))\
                                                               .transform_to(coord.AltAz(location=coord.EarthLocation(lon=longitude * u.deg, lat=latitude * u.deg),\
                                                                                         obstime=Time(x, format='iso', scale='utc')
                                                                                        )
                                                                            ).zen.degree
                                                              ).values


df_resample.to_csv(dir_out+node_id+'.csv')
