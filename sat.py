# -*- coding: utf-8 -*-
#
# Description: Utilities to work with satellite data.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['wind_subset',
           'dl_goes',
           'MUR_data']

import os
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from datetime import datetime,timedelta
from ap_tools.utils import lon360to180,lon180to360
from pandas import Panel

def wind_subset(times=(datetime(2009,12,28), datetime(2010,1,10)),
    dt='daily', llcrnrlon=-45, urcrnrlon=-35, llcrnrlat=-25, urcrnrlat=-18):
    """
    USAGE
    -----
    u,v = wind_subset(...)

    Gets wind vectors from the NCDC/NOAA Blended Seawinds L4 product,
    given a given lon, lat, time bounding box.

    Input
    -----
    * times: A tuple with 2 datetime objects marking the
      start and end of the desired subset.
      If dt=='clim', this should be an integer indicating the
      desired month, from 1 (January) through 12 (December).
      This climatology is built from the 1995-2005 data.

    * dt: Time resolution wanted. Choose `six-hourly`,
      `daily` (default) or `monthly`.

    * llcrnrlon, urcrnrlon: Longitude band wanted.
    * llcrnrlat, urcrnrlat: Latitude band wanted.

    Returns
    -------
    u,v: pandas Panels containing the data subsets.

    Example
    -------
    TODO.
    """
    head = 'http://www.ncdc.noaa.gov/thredds/dodsC/oceanwinds'
    tail = {'six-hourly':'6hr','daily':'dly','monthly':'mon','clim':'clm-agg'}

    url = head + tail[dt]
    nc = Dataset(url)

    lon = nc.variables.pop('lon')[:]
    lon = lon360to180(lon) # Longitude is in the [0,360] range, BB is in the [-180,+180] range.
    lat = nc.variables.pop('lat')[:]
    time = nc.variables.pop('time')
    time = num2date(time[:],time.units,calendar='standard')

    # Subsetting the data.
    maskx = np.logical_and(lon >= llcrnrlon, lon <= urcrnrlon)
    masky = np.logical_and(lat >= llcrnrlat, lat <= urcrnrlat)

    # Ignoring more precise time units than
    # the resolution in the desired dataset.
    nt = time.size
    if dt=='six-hourly':
        for i in range(nt):
            time[i] = time[i].replace(minute=0,second=0)
    elif dt=='daily':
        for i in range(nt):
            time[i] = time[i].replace(hour=0,minute=0,second=0)
    elif dt=='monthly':
        for i in range(nt):
            time[i] = time[i].replace(day=1,hour=0,minute=0,second=0)
    elif dt=='clim':
        time[1] = time[1].replace(month=2) # Fixing February.
        aux = []; [aux.append(Time.month) for Time in time]
        time = np.copy(aux); del aux
    else:
        pass

    # If the time wanted is a single month/day/hour,
    # get the nearest time available in the dataset.
    if dt=='clim':
        maskt=time==times
    else:
        if np.size(times)<2:
            idxt = np.abs(time-times).argmin()
            maskt=time==time[idxt]
        else:
            maskt = np.logical_and(time >= times[0], time <= times[1])

    np.disp('Downloading subset...')
    lon, lat, time = lon[maskx], lat[masky], time[maskt]
    u = nc.variables['u'][maskt,0,masky,maskx]
    v = nc.variables['v'][maskt,0,masky,maskx]
    np.disp('Downloaded.')

    lat,lon,time,u,v = map(np.atleast_1d, [lat,lon,time,u,v])

    # Sets the panels.
    U = Panel(data=u, items=time, major_axis=lat, minor_axis=lon)
    V = Panel(data=v, items=time, major_axis=lat, minor_axis=lon)

    # Sorting major axis (longitude) to fix the crossing of the 0º meridian.
    U = U.sort_index(axis=2)
    V = V.sort_index(axis=2)

    # Retrieves the u,v arrays to fix masked values.
    u,v,lon,lat = map(np.atleast_1d, (U.values,V.values,U.minor_axis.values,U.major_axis.values))
    fill_value = -9999.
    u = np.ma.masked_equal(u, fill_value)
    v = np.ma.masked_equal(v, fill_value)

    # Resets the Panels with the sorted and masked u,v arrays.
    U = Panel(data=u, items=time, major_axis=lat, minor_axis=lon)
    V = Panel(data=v, items=time, major_axis=lat, minor_axis=lon)

    return U,V

def dl_goes(time=datetime(2013,9,13), dt=24, dest_dir='./'):
    """
    USAGE
    -----
    dl_goes(time=datetime(2013,9,13))

    Downloads full GOES SST images from the PODAAC FTP (12 MB each). Uses wget.

    ftp://podaac-ftp.jpl.nasa.gov/allData/goes/L3/goes_6km_nrt/americas/.

    * `time` is a datetime object or a list of datetime objects containing the desired times.

    * `dt` is an integer representing the time resolution desired. Choose from `24` (default),
    `3` or `1`. If `dt` is either 1 or 3, all images for each day in `time` will be downloaded.

    * `dest_dir` is the directory in which the downloaded data will be saved.

    TODO
    ----
    Find an openDAP link for this dataset (WITH the bayesian cloud filter).
    """
    if type(time)!=list:
        time = [time]

    original_dir = os.getcwd()  # Store original working directory.
    if os.path.isdir(dest_dir): # If dest_dir already exists.
        os.chdir(dest_dir)
    else:                       # Create it if it does not exist.
        os.makedirs(dest_dir)
        os.chdir(dest_dir)

    for date in time: # Getting files for each day in the list.
        yyyy = str(date.year)
        dd = date.timetuple().tm_yday # Get the julian day.
        dd = str(dd).zfill(3)
        head = 'ftp://podaac-ftp.jpl.nasa.gov/OceanTemperature/goes/L3/goes_6km_nrt/americas/%s/%s/' %(yyyy,dd)
        filename = 'sst%s?_%s_%s' % (str(dt),yyyy,dd) # dt can be 1, 3 or 24 (hourly, 3-hourly or daily).
        url = head + filename                         # The 'b' character is only for 2008-present data.
        cmd = "wget --tries=inf %s" %url
        os.system(cmd) # Download file.

    os.chdir(original_dir) # Return to the original working directory.
    np.disp("Done downloading all files.")

    return None

class MUR_data(object):
    """
    USAGE
    -----
    mur = MUR_data(filename)

    A simple container class for MUR data (http://mur.jpl.nasa.gov/).
    """
    def __init__(self, filename):
        self.ncfile = Dataset(filename)
        self.varlist = list(self.ncfile.variables)
        self.lon = self.ncfile.variables['lon'][:]
        self.lat = self.ncfile.variables['lat'][:]
        self.sst = self.ncfile.variables['analysed_sst'][:]
        self.ncfile.close()
        self.x, self.y = np.meshgrid(self.lon, self.lat)
        self.sst = self.sst.squeeze()
        self.sst -= 273.15
