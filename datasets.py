# Description: Functions to work with datasets.
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['topo_subset',
           'get_indices']

import numpy as np
from netCDF4 import Dataset
from ap_tools.utils import lon360to180

def topo_subset(llcrnrlon=-42, urcrnrlon=-35, llcrnrlat=-23,
                urcrnrlat=-14, tfile='topo30.grd'):
    """
    Get a subset from an etopo1, etopo2 or Smith and Sandwell topography file.

    OBS: Modified from oceans.datasets.etopo_subset() function by Filipe Fernandes (ocefpaf@gmail.com).
    """
    topo = Dataset(tfile, 'r')

    if 'smith_sandwell' in tfile.lower() or 'etopo1' in tfile.lower():
        lons = topo.variables["lon"][:]
        lats = topo.variables["lat"][:]
        lons = lon360to180(lons)
    elif 'etopo2' in tfile.lower():
        lons = topo.variables["x"][:]
        lats = topo.variables["y"][:]
    else:
        np.disp('Unknown topography file.')
        return

    res = get_indices(llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, lons, lats)
    lon, lat = np.meshgrid(lons[res[0]:res[1]], lats[res[2]:res[3]])
    bathy = topo.variables["z"][int(res[2]):int(res[3]),
                                int(res[0]):int(res[1])]

    return lon, lat, bathy

def get_indices(min_lat, max_lat, min_lon, max_lon, lons, lats):
    """
    Return the data indices for a lon, lat square.

    From Filipe Fernandes' (ocefpaf@gmail.com) oceans module (https://github.com/ocefpaf/python-oceans).
    """
    distances1, distances2, indices = [], [], []
    index = 1
    for point in lats:
        s1 = max_lat - point
        s2 = min_lat - point
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index - 1))
        index = index + 1

    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    distances1, distances2 = [], []
    index = 1
    for point in lons:
        s1 = max_lon - point
        s2 = min_lon - point
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index - 1))
        index = index + 1

    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])

    # max_lat_indices, min_lat_indices, max_lon_indices, min_lon_indices.
    res = np.zeros((4), dtype=np.float64)
    res[0] = indices[3][2]
    res[1] = indices[2][2]
    res[2] = indices[1][2]
    res[3] = indices[0][2]

    return res
