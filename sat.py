# Description: Utilities to work with satellite data.
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['dl_goes',
           'MUR_data']

import os
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from datetime import datetime,timedelta
from ap_tools.utils import lon360to180,lon180to360

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

    A simple container class for MUR SST data (http://mur.jpl.nasa.gov/).
    """
    def __init__(self, filename):
        self.ncfile = Dataset(filename)
        self.varlist = list(self.ncfile.variables)
        self.lon = self.ncfile.variables['lon'][:]
        self.lat = self.ncfile.variables['lat'][:]
        t = self.ncfile.variables['time']
        self.time = num2date(t[:], units=t.units)
        self.sst = self.ncfile.variables['analysed_sst'][:]
        self.ncfile.close()
        self.x, self.y = np.meshgrid(self.lon, self.lat)
        self.sst = self.sst.squeeze()
        self.sst -= 273.15 # K to degrees C.
