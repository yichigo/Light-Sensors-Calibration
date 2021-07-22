import pandas as pd
import numpy as np
import pytz
import os
import sys

node_id = '001e06305a6b'
dir_out = '../lightsensors/'

def main(argv):
    if argv != None:
        node_id = argv[1]

    dir_in = '/Volumes/MINTS/raw/'+ node_id + '/'

    df_lightsensors = pd.DataFrame()

    year_list = os.listdir(dir_in)
    year_list.sort()
    year_list = ['2019','2020']

    for year in year_list: # year
        dir_year = dir_in + year
        if os.path.isfile(dir_year):
            continue
        print(dir_year)
        
        month_list = os.listdir(dir_year)
        month_list.sort()

        if year == '2019':
            month_list = ['12']
        elif year == '2020':
            month_list = ['01','02','03','04']
        else:
            continue
        print(month_list)

        for month in month_list: # month
            dir_month = dir_year + '/' + month
            if os.path.isfile(dir_month):
                continue
            print(dir_month)

            day_list = os.listdir(dir_month)
            day_list.sort()

            for day in day_list: # day
                dir_day = dir_month + '/' + day
                if os.path.isfile(dir_day):
                    continue
                print(year, month, day)
                fn_list = os.listdir(dir_day)
                if len(fn_list)<11: ###   Perhaps Delete this 
                    continue
                fn_list.sort()
                
                # Firstly, merge SKYCAM: every 2 minutes
                for fn in fn_list:
                    fn_in = dir_day + '/' + fn
                    if  os.path.isdir(fn_in):
                        continue
                    sensor_name = fn[19:-15]
                    if sensor_name == 'SKYCAM_002':
                        data = pd.read_csv(fn_in, parse_dates=True, index_col = 'dateTime')
                        data = data.resample('10s').mean()
                        data_day = data

                for fn in fn_list:
                    if fn[0] == '.':
                        continue
                    fn_in = dir_day + '/' + fn
                    if os.path.isdir(fn_in):
                        continue
                        
                    sensor_name = fn[19:-15]
                    if sensor_name =='SKYCAM_002':# or sensor_name =='TSL2591':#'SKYCAM':
                        continue
                    
                    data = pd.read_csv(fn_in, parse_dates=True, index_col = 'dateTime')
                    data = data.resample('10S').mean()
                    #data.index = pd.to_datetime(data.index)

                    # remove ' ' in columns name
                    var_list = []
                    for var in data.columns:
                        var = var.replace(' ','')
                        var_list.append(var)
                    data.columns = var_list
                    
                    var_list = []
                    var_names = []
                    if (sensor_name == 'AS7262'): ### redundancy removed
                        var_list = ['violetPre','bluePre','greenPre','yellowPre','orangePre','redPre']#,'violetCalibrated','blueCalibrated','greenCalibrated','yellowCalibrated','orangeCalibrated','redCalibrated']
                        var_names = ['Violet','Blue','Green','Yellow','Orange','Red']
                    elif (sensor_name == 'BME280'):
                        var_list = ['temperature','pressure','humidity']
                        var_names = ['Temperature','Pressure','Humidity']
                    elif (sensor_name == 'GPSGPGGA2'):
                        var_list = ['latitudeCoordinate','longitudeCoordinate','altitude']
                        var_names = ['Latitude','Longitude','Altitude']
                    elif (sensor_name == 'GPSGPRMC2'):
                        var_list = []
                        var_names = []
                    elif (sensor_name == 'MGS001'):
                        var_list = ['nh3','co','no2','c3h8','c4h10','ch4','h2','c2h5oh']
                        var_names = ['NH3','CO','NO2','C3H8','C4H10','CH4','H2','C2H5OH']
                    #elif (sensor_name == 'OPCN2'): # drop this table
                    #    var_list = ['binCount0','binCount1','binCount2','binCount3','binCount4','binCount5','binCount6','binCount7','binCount8','binCount9','binCount10','binCount11','binCount12','binCount13','binCount14','binCount15','bin1TimeToCross','bin3TimeToCross','bin5TimeToCross','bin7TimeToCross','pm1', 'pm2_5', 'pm10']
                    #    var_names = ['binCount0','binCount1','binCount2','binCount3','binCount4','binCount5','binCount6','binCount7','binCount8','binCount9','binCount10','binCount11','binCount12','binCount13','binCount14','binCount15','bin1TimeToCross','bin3TimeToCross','bin5TimeToCross','bin7TimeToCross','pm1', 'pm2.5', 'pm10']
                    #elif (sensor_name == 'PPD42NSDuo'):  data frequency is too low, drop this table
                    #    var_list = ['LPOPmMid','LPOPm10','ratioPmMid','ratioPm10','concentrationPmMid','concentrationPm2_5','concentrationPm10']
                    #    var_names = ['LPOPmMid','LPOPm10','ratioPmMid','ratioPm10','concentrationPmMid','concentrationPm2_5','concentrationPm10']
                    elif (sensor_name == 'SCD30'):
                        var_list =['c02'] # this is a typo in data file
                        var_names = ['c02']
                    elif (sensor_name == 'TSL2591'):  ### perhaps redundant
                        var_list = ['luminosity','ir','full','visible', 'lux'] # lux is not NaN now # lux might be NaN
                        var_names = ['Luminosity','IR','Full','Visible', 'Lux']
                    elif (sensor_name == 'VEML6075'): ### redundancy removed
                        var_list = ['rawUVA','rawUVB','visibleCompensation','irCompensation','index']
                        var_names = ['UVA','UVB','Visible Compensation','IR Compensation','UV Index']
                    
                    for var in var_list:
                        if type(data[[var]].values[0][0])==str: # when there is abnormal values, type becomes str
                            data[[var]] = np.array(data[[var]], dtype = float)
                    
                    data_add = data[var_list]
                    data_add.columns = var_names
                    data_day = pd.concat([data_day, data_add], axis=1)
                    
                df_lightsensors = pd.concat([df_lightsensors, data_day])

    df_lightsensors.index.name = 'UTC'
    print("data length: ", len(df_lightsensors))

    print("dropna for all NaN")
    df_lightsensors.dropna(how = "all", inplace = True)
    print("data length: ", len(df_lightsensors))

    print("drop_duplicates")
    df_lightsensors.drop_duplicates(inplace=True)
    print("data length: ", len(df_lightsensors))

    fn = dir_out + 'MINTS_'+node_id+'.csv'
    df_lightsensors.to_csv(fn)
    print(df_lightsensors.head())


if __name__ == '__main__':
    main(sys.argv)


