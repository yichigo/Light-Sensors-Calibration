import numpy as np
import pandas as pd
import math
# import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyart

# from pysolar.solar import get_altitude, get_azimuth, constants
# from pysolar.solar import get_azimuth, get_sun_earth_distance, get_projected_radial_distance
# from pysolar import solartime as stime
# import pytz

import cartopy.crs as ccrs

import os
import time
import datetime

#  32.5,  33.5
# -97.5, -96.5

import warnings
warnings.filterwarnings(action='ignore')



folder = "spatial"
node_id = "10004098"
dir_NEXRAD = '/Volumes/Backup Plus/NEXRAD/data/'
dir_data = "../data/"
dir_out = "../figures/" + folder + "/"

fn_in = dir_data + "driving_" + node_id + ".csv"
df = pd.read_csv(fn_in, parse_dates=True, index_col = 'UTC')
df.head()


# average for logarithm dbz, do average on Z
def log10_mean(dbz1, dbz2, w1=0.5, w2=0.5):
    return 10*np.log10(w1*10**(dbz1/10) + w2*10**(dbz2/10))

# get NEXRAD filenames on a date
def get_filenames_by_date(date):
    dir_date = dir_NEXRAD + str(date) + '/'
    filenames = os.listdir(dir_date)
    filenames = [dir_date + filename for filename in filenames if (filename[0]!='.') & (filename[-1]!='M')]
    return sorted(filenames)



# NEXRAD variables
variables = ['cross_correlation_ratio',
             'differential_phase',
             'differential_reflectivity',
             'reflectivity',
             'spectrum_width',
             'velocity',
             'ROI']

fill_values = {'cross_correlation_ratio':0,
               'differential_phase':0,
               'differential_reflectivity':0,
               'reflectivity':-9999, # since logarithm, dbz = 10 log_10(Z/Z_0)
               'spectrum_width':0,
               'velocity':0}

# NEXRAD parameters, from an arbitrary NEXRAD file
filename = '/Volumes/Backup Plus/NEXRAD/data/2020-02-10/2020_02_10_KFWS_KFWS20200210_140005_V06'
radar = pyart.io.read(filename)
radar_longitude = radar.longitude['data'][0]
radar_latitude = radar.latitude['data'][0]
radar_altitude = radar.altitude['data'][0]
projparams = {'proj': 'pyart_aeqd',
             'lon_0': radar_longitude,
             'lat_0': radar_latitude}

# grid range around the car
delta_x, delta_y, delta_z = 20000, 20000, 10000 # in meters
num_x, num_y, num_z = 401, 401, 11

# matrix index of grid center 
col_c, row_c = int((num_x-1)/2), int((num_y-1)/2)

# grid resolution
dx = 2*delta_x/(num_x-1) # 100 m
dy = 2*delta_y/(num_y-1) # 100 m
dz = delta_z/(num_z-1) # 1000 m

# return x-y 3x3 grid: [N,NE,E,SE,S,SW,W,NW] by 5000m, z 11 grid: 0 - 10*1000m
# in the matrix form, the x-y direction is [[ SW, W, SE],
#                                           [  W, C,  E],
#                                           [ NW, N, NE]]
directions = ['SW', 'S', 'SE',  'W', '', 'E', 'NW', 'N', 'NE']
dcol = int(5000 / dx)
drow = int(5000 / dy)


heights = [str(i) + 'km' for i in range(num_z)]
heights_directions = [ height + ' '*bool(direction) + direction for height in heights for direction in directions]
variables_heights_directions = {var: [(var +' '+ temp) for temp in heights_directions] for var in variables }
for var in variables:
    for variable in variables_heights_directions[var]:
        df[variable] = None

# loop by date
dates = sorted(set(df.index.date))
# initialize fn_prev to the last file name of yesterday's folder
fn_prev = get_filenames_by_date(dates[0])[0]
time_prev = datetime.datetime.strptime(fn_prev[-19:-4], '%Y%m%d_%H%M%S')
radar_prev = None




start_time = time.time()
for date in dates:
    print(date)
    
    date_str = str(date)
    dir_date = dir_NEXRAD + date_str + '/'
    
    fns = get_filenames_by_date(date)
    # merge NEXRAD data into df
    i = 0
    while i < len(fns):
        fn_curr = fns[i]
        time_curr = datetime.datetime.strptime(fn_curr[-19:-4], '%Y%m%d_%H%M%S')
        indices = df.index[(df.index >= time_prev) & (df.index <= time_curr)]
        time_delta = (time_curr-time_prev)
        
        if len(indices) == 0:
            # use a None radar_curr for the next radar_prev
            radar_curr = None
        else:
            print(time_curr, ' ', len(indices), '/', len(df))
            # read NEXRAD data of time_prev and time_curr
            if not radar_prev:
                radar_prev = pyart.io.read(fn_prev)
            radar_curr = pyart.io.read(fn_curr)
            
            # average car position during (time_prev, time_curr)
            longitude, latitude, altitude = df.loc[indices, ['longitude','latitude','altitude']].mean()
            
            # average car position in x, y, z
            x_c, y_c = pyart.core.geographic_to_cartesian(longitude, latitude, projparams)
            x_c, y_c = x_c[0], y_c[0]
            z_c = altitude - radar_altitude
            
            # gridize NEXRAD data around car
            grid_limits=((z_c, z_c + delta_z),
                         (x_c - delta_x, x_c + delta_x),
                         (y_c - delta_y, y_c + delta_y))
            grid_prev = pyart.map.grid_from_radars(radar_prev, grid_shape=(num_z, num_y, num_x), grid_limits = grid_limits)
            grid_curr = pyart.map.grid_from_radars(radar_curr, grid_shape=(num_z, num_y, num_x), grid_limits = grid_limits)
            
            # merge NEXRAD data between time_prev and time_curr
            for time_car in indices:
                # weights of radar_prev and radar_curr
                weight_prev = (time_curr - time_car)/time_delta
                weight_curr = (time_car - time_prev)/time_delta
                
                # car position
                longitude_car, latitude_car = df.loc[time_car, ['longitude','latitude']]
                x_car, y_car = pyart.core.geographic_to_cartesian(longitude_car, latitude_car, projparams)
                x_car = x_car[0]
                y_car = y_car[0]
                col, row = col_c + round((x_car-x_c)/dx), row_c + round((y_car-y_c)/dy)
                
                # some var may not exist on some date
                for var in grid_prev.fields.keys() & grid_curr.fields.keys():
                    data_prev = grid_prev.fields[var]['data'][:, row-drow:row+drow+1:drow, col-dcol:col+dcol+1:dcol]
                    data_curr = grid_curr.fields[var]['data'][:, row-drow:row+drow+1:drow, col-dcol:col+dcol+1:dcol]
                    
                    # fillna
                    if var != 'ROI':
                        data_prev = data_prev.filled(fill_value=fill_values[var])
                        data_curr = data_curr.filled(fill_value=fill_values[var])
                    
                    # weighted interpolation
                    if var == 'reflectivity':
                        data_var = log10_mean(data_prev, data_curr, weight_prev, weight_curr) # interpolation on Z, not dbz
                    else:
                        data_var = weight_prev * data_prev + weight_curr * data_curr
                    
                    df.loc[time_car, variables_heights_directions[var]] = data_var.flatten()
            
            print('cost time: %s seconds ' % (time.time() - start_time))
            
        fn_prev = fn_curr
        time_prev = time_curr
        radar_prev = radar_curr
        i += 1
        

fn_out = dir_data + "driving_" + node_id + "_NEXRAD.csv"
df.to_csv(fn_out)