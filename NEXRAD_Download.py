import numpy as np
import datetime
import boto
import os
import time
#suppress deprecation warnings
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
from pathlib import Path
import glob
import boto3

########################################################################################################
# Function to conpare the difference in two lists
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1]
    return li_dif

# Get all sites' name
def get_allsites():
    all_sites = []
    locs = pyart.io.nexrad_common.NEXRAD_LOCATIONS
    for key in locs:
        all_sites.append(key)
    return all_sites

########################################################################################################
def customized_download(date='2020/05/16',all_sites=['KFWS']):

    #create a datetime object for the current time in UTC and use the
    # year, month, and day to drill down into the NEXRAD directory structure.
    #date = ("{:4d}".format(now.year) + '/' + "{:02d}".format(now.month) + '/' +
    #        "{:2d}".format(now.day) + '/')


    print("Search Date %s" %(date))

    awscount = 0
    Fail_lst =[]
    awslst = []
    #get the bucket list for the selected date

    #use boto to connect to the AWS nexrad holdings directory
    s3conn = boto.connect_s3()
    bucket = s3conn.get_bucket('noaa-nexrad-level2')
    s3 = boto3.resource('s3')

    #Note: this returns a list of all of the radar sites with data for
    # the selected date
    ls = bucket.list(prefix=date + '/',delimiter='/')
    for item in ls:
        awslst.append(item.name.split('/')[-2])

    #Find the Missing sites from AWS lst at the select date
    li3 = Diff(awslst, all_sites)
    print("Missing sites : %s " %(li3))
    
    for key in ls:
        print(key.name)
        awscount+=1
    print("%d sites are selected, total %d sites returned from AWS at %s %s" %(len(all_sites),awscount-1,date,time))
    print("===================================================================================")
    print('\n')

    for site in all_sites:
        for key in ls:
            #only pull the data and save the arrays for the site we want
            if site in key.name.split('/')[-2]:
                print("%s has been found in AWS return list " %(site))
                #set up the path to the NEXRAD files
                path = date +'/' + site + '/' + site

                keys = bucket.get_all_keys(prefix=path)
                
                n = 1
                for s3key in keys:
                    try: 
                        print("Downloading %s (%d/%d)" %(s3key.name,n,len(keys)))
                        Path(os.path.join(date.replace("/","-"))).mkdir(parents = True, exist_ok=True)
                        s3.Bucket('noaa-nexrad-level2').download_file(s3key.name, os.path.join(date.replace("/","-"),s3key.name.replace("/","_")))
                        n += 1
                    except:
                        print("%s not read sucsseful <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" %(s3key.name))
                        Fail_lst.append(s3key.name)
                        print('\n')
                        
        print("Failed reading sites %s" %(Fail_lst))
        
        
        
#####################################################################################################
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + datetime.timedelta(n)

##################################################################################################



if __name__ == "__main__":
    # define a spacific date
#     mydate = "2020/06/24"
#     mysites = ['KFWS']
#     customized_download(date=mydate,all_sites = mysites)
    
    
    # Define a date range

    mysites = ['KFWS']
    start_dt = datetime.date(2020,6, 25)  # custommized your starting date and end date here
    end_dt = datetime.date(2020,6,27)
    for dt in daterange(start_dt, end_dt):
        customized_download(date=dt.strftime("%Y/%m/%d"),all_sites = mysites)
    