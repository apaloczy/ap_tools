# Description: General-purpose functions for personal use.
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['flowfun',
           'cumsimp',
           'rot_vec',
           'avgdir',
           'lon180to360',
           'lon360to180',
           'bbox2ij',
           'get_arrdepth',
           'fpointsbox',
           'near',
           'near2',
           'mnear',
           'refine',
           'denan',
           'standardize',
           'linear_trend',
           'thomas',
           'point_in_poly',
           'get_mask_from_poly',
           'sphericalpolygon_area',
           'greatCircleBearing',
           'weim',
           'smoo2',
           'topo_slope',
           'curvature_geometric',
           'get_isobath',
           'angle_isobath',
           'isopyc_depth',
		   'whiten_zero',
           'wind2stress',
           'gen_dates',
           'fmt_isobath',
           'float2latex',
           'mat2npz',
           'bb_map',
           'dots_dualcolor']

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import path
from mpl_toolkits.basemap import Basemap
from datetime import datetime,timedelta
from dateutil import rrule,parser
from scipy.io import loadmat,savemat
from glob import glob
from netCDF4 import Dataset, num2date
from pandas import Panel
from gsw import distance

def flowfun(x, y, u, v, variable='psi', geographic=True):
	"""
	FLOWFUN  Computes the potential PHI and the streamfunction PSI
	 of a 2-dimensional flow defined by the matrices of velocity
	 components U and V, so that

	       d(PHI)    d(PSI)          d(PHI)    d(PSI)
	  u =  -----  -  ----- ,    v =  -----  +  -----
	        dx        dy              dx        dy

	P = FLOWFUN(x,y,u,v) returns an array P of the same size as u and v,
	which can be the velocity potential (PHI) or the streamfunction (PSI)
	Because these scalar fields are defined up to the integration constant,
	their absolute values are such that PHI[0,0] = PSI[0,0] = 0.

	For a potential (irrotational) flow  PSI = 0, and the Laplacian
	of PSI is equal to the divergence of the velocity field.

	A solenoidal (non-divergent) flow can be described by the
	streamfunction alone, and the Laplacian of the streamfunction
	is equal to the vorticity (curl) of the velocity field.

	The units of the grid coordinates are assumed to be consistent
	with the units of the velocity components, e.g., [m] and [m/s].

	If variable=='psi', the streamfunction (PSI) is returned.

	If variable=='phi', the velocity potential (PHI) is returned.

	If geographic==True (default), (x,y) are assumed to be
	(longitude,latitude) and are converted to meters before
	computing (dx,dy).

	If geographic==False, (x,y) are assumed to be in meters.

	Uses function 'cumsimp()' (Simpson rule summation).

	Author: Kirill K. Pankratov, March 7, 1994.
	Source: http://www-pord.ucsd.edu/~matlab/stream.htm
	Translated to Python by André Palóczy, January 15, 2015.
	Modified by André Palóczy on January 15, 2015.
	"""
	x,y,u,v = map(np.asanyarray, (x,y,u,v))

	if not x.shape==y.shape==u.shape==v.shape:
		print("Error: Arrays (x, y, u, v) must be of equal shape.")
		return

	## Calculating grid spacings.
	if geographic:
		dlat, _ = np.gradient(y)
		_, dlon = np.gradient(x)
		deg2m = 111120.0                     # [m/deg]
		dx = dlon*deg2m*np.cos(y*np.pi/180.) # [m]
		dy = dlat*deg2m                      # [m]
	else:
		dy, _ = np.gradient(y)
		_, dx = np.gradient(x)

	ly, lx = x.shape                         # Shape of the (x,y,u,v) arrays.

	## Now the main computations.
	## Integrate velocity fields to get potential and streamfunction.
	## Use Simpson rule summation (function CUMSIMP).

	## Compute velocity potential PHI (non-rotating part).
	if variable=='phi':
		cx = cumsimp(u[0,:]*dx[0,:])         # Compute x-integration constant
		cy = cumsimp(v[:,0]*dy[:,0])         # Compute y-integration constant
		cx = np.expand_dims(cx, 0)
		cy = np.expand_dims(cy, 1)
		phiy = cumsimp(v*dy) + np.tile(cx, (ly,1))
		phix = cumsimp(u.T*dx.T).T + np.tile(cy, (1,lx))
		phi = (phix + phiy)/2.
		return phi

	## Compute streamfunction PSI (non-divergent part).
	if variable=='psi':
		cx = cumsimp(v[0,:]*dx[0,:])         # Compute x-integration constant
		cy = cumsimp(u[:,0]*dy[:,0])         # Compute y-integration constant
		cx = np.expand_dims(cx, 0)
		cy = np.expand_dims(cy, 1)
		psix = -cumsimp(u*dy) + np.tile(cx, (ly,1))
		psiy = cumsimp(v.T*dx.T).T - np.tile(cy, (1,lx))
		psi = (psix + psiy)/2.
		return psi

def cumsimp(y):
	"""
	F = CUMSIMP(Y)    Simpson-rule column-wise cumulative summation.
	Numerical approximation of a function F(x) such that
	Y(X) = dF/dX.  Each column of the input matrix Y represents
	the value of the integrand  Y(X)  at equally spaced points
	X = 0,1,...size(Y,1).
	The output is a matrix  F of the same size as Y.
	The first row of F is equal to zero and each following row
	is the approximation of the integral of each column of matrix
	Y up to the givem row.
	CUMSIMP assumes continuity of each column of the function Y(X)
	and uses Simpson rule summation.
	Similar to the command F = CUMSUM(Y), exept for zero first
	row and more accurate summation (under the assumption of
	continuous integrand Y(X)).

	Author: Kirill K. Pankratov, March 7, 1994.
	Source: http://www-pord.ucsd.edu/~matlab/stream.htm
	Translated to Python by André Palóczy, January 15, 2015.
	"""
	y = np.asanyarray(y)

	## 3-point interpolation coefficients to midpoints.
	## Second-order polynomial (parabolic) interpolation coefficients
	## from  Xbasis = [0 1 2]  to  Xint = [.5 1.5]
	c1 = 3/8.
	c2 = 6/8.
	c3 = -1/8.

	if y.ndim==1:
		y = np.expand_dims(y,1)
		f = np.zeros((y.size,1))    # Initialize summation array.
		squeeze_after = True
	elif y.ndim==2:
		f = np.zeros(y.shape)       # Initialize summation array.
		squeeze_after = False
	else:
		print("Error: Input array has more than 2 dimensions.")
		return

	if y.size==2:                   # If only 2 elements in columns - simple average.
		f[1,:] = (y[0,:] + y[1,:])/2.
		return f
	else:                           # If more than two elements in columns - Simpson summation.
		## Interpolate values of y to all midpoints.
		f[1:-1,:] = c1*y[:-2,:] + c2*y[1:-1,:] + c3*y[2:,:]
		f[2:,:] = f[2:,:] + c3*y[:-2,:] + c2*y[1:-1,:] + c1*y[2:,:]
		f[1,:] = f[1,:]*2
		f[-1,:] = f[-1,:]*2

		## Simpson (1,4,1) rule.
		f[1:,:] = 2*f[1:,:] + y[:-1,:] + y[1:,:]
		f = np.cumsum(f, axis=0)/6. # Cumulative sum, 6 - denominator from the Simpson rule.

	if squeeze_after:
		f = f.squeeze()

	return f

def rot_vec(u, v, angle=-45, degrees=True):
	"""
	USAGE
	-----
	u_rot,v_rot = rot_vec(u,v,angle=-45.,degrees=True)

	Returns the rotated vector components (`u_rot`,`v_rot`)
	from the zonal-meridional input vector components (`u`,`v`).
	The rotation is done using the angle `angle` positive counterclockwise
	(trigonometric convention). If `degrees` is set to `True``(default),
	then `angle` is converted to radians.
	is

	Example
	-------
	>>> from matplotlib.pyplot import quiver
	>>> from ap_tools.utils import rot_vec
	>>> u = -1.
	>>> v = -1.
	>>> u2,v2 = rot_vec(u,v, angle=-30.)
	"""
	u,v = map(np.asanyarray, (u,v))
	if degrees:
		angle = angle*np.pi/180. # Degrees to radians.

	u_rot = +u*np.cos(angle) + v*np.sin(angle) # Usually the across-shore component.
	v_rot = -u*np.sin(angle) + v*np.cos(angle) # Usually the along-shore component.

	return u_rot,v_rot

def avgdir(dirs, degrees=False, axis=None):
    """
    USAGE
    -----
    dirm = avgdir(dirs, degrees=False, axis=None)

    Calculate the mean direction of an array of directions 'dirs'.
    If 'degrees' is 'False' (default), the input directions must be
    in radians. If 'degrees' is 'True', the input directions must be
    in degrees.

    The direction angle is measured from the ZONAL axis, i.e.,
    (0, 90, -90) deg are (Eastward, Northward, Southward).
    180 and -180 deg are both Westward.

    If 'axis' is 'None' (default) the mean is calculated on the
    flattened array. Otherwise, 'axis' is the index of the axis
    to calculate the mean over.
    """
    dirs = np.array(dirs)

    if degrees:
        dirs = dirs*np.pi/180 # Degrees to radians.

    uxs = np.cos(dirs)
    vys = np.sin(dirs)
    dirm = np.arctan2(vys.sum(axis=axis), uxs.sum(axis=axis))

    if degrees:
        dirm = dirm*180/np.pi # From radians to degrees.

    return dirm

def lon180to360(lon):
	"""
	Converts longitude values in the range [-180,+180]
	to longitude values in the range [0,360].
	"""
	lon = np.asanyarray(lon)
	return (lon + 360.0) % 360.0

def lon360to180(lon):
	"""
	Converts longitude values in the range [0,360]
	to longitude values in the range [-180,+180].
	"""
	lon = np.asanyarray(lon)
	return ((lon + 180.) % 360.) - 180.

def bbox2ij(lon, lat, bbox=[-135., -85., -76., -64.], FIX_IDL=True):
    """
	USAGE
	-----
	ilon_start, ilon_end, jlat_start, jlat_end = bbox2ij(lon, lat, bbox=[-135., -85., -76., -64.], FIX_IDL=True)

    OR

    (ilon_start_left, ilon_end_left, jlat_start, jlat_end), (ilon_start_right, ilon_end_right, jlat_start, jlat_end) = ...
    ... bbox2ij(lon, lat, bbox=[-135., -85., -76., -64.], FIX_IDL=True)

    Return indices for i,j that will completely cover the specified bounding box. 'lon' and 'lat' are 2D coordinate arrays
    (generated by meshgrid), and 'bbox' is a list like [lon_start, lon_end, lat_start, lat_end] describing the desired
    longitude-latitude box.

    If the specified bbox is such that it crosses the edges of the longitude array, two tuples of indices are returned.
    The first (second) tuple traces out the left (right) part of the bbox.

    If FIX_IDL is set to 'True' (default), the indices returned correspond to the "short route" around the globe, which
    amounts to assuming that the specified bbox crosses the International Date. If FIX_IDL is set to 'False', the
    "long route" is used instead.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> lon = np.arange(-180., 180.25, 0.25)
    >>> lat = np.arange(-90., 90.25, 0.25)
    >>> lon, lat = np.meshgrid(lon, lat)
    >>> h = np.sin(lon) + np.cos(lat)
    >>> i0, i1, j0, j1 = bbox2ij(lon, lat, bbox=[-71, -63., 39., 46])
    >>> h_subset = h[j0:j1,i0:i1]
    >>> lon_subset = lon[j0:j1,i0:i1]
    >>> lat_subset = lat[j0:j1,i0:i1]
    >>> fig, ax = plt.subplots()
    >>> ax.pcolor(lon_subset,lat_subset,h_subset)
    >>> plt.axis('tight')

    Original function downloaded from http://gis.stackexchange.com/questions/71630/subsetting-a-curvilinear-netcdf-file-roms-model-output-using-a-lon-lat-boundin
    Modified by André Palóczy on August 20, 2016 to handle bboxes that
    cross the International Date Line or the edges of the longitude array.
    """
    lon, lat, bbox = map(np.asanyarray, (lon, lat, bbox))

    # Test whether the wanted bbox crosses the International Date Line (brach cut of the longitude array).
    dlon = bbox[:2].ptp()
    IDL_BBOX=dlon>180.
    IDL_BBOX=np.logical_and(IDL_BBOX, FIX_IDL)

    mypath = np.array([bbox[[0,1,1,0]], bbox[[2,2,3,3]]]).T
    p = path.Path(mypath)
    points = np.vstack((lon.flatten(), lat.flatten())).T
    n, m = lon.shape
    inside = p.contains_points(points).reshape((n, m))

    # Fix mask if bbox goes throught the International Date Line.
    if IDL_BBOX:
        fcol=np.all(~inside, axis=0)
        flin=np.any(inside, axis=1)
        fcol, flin = map(np.expand_dims, (fcol, flin), (0, 1))
        fcol = np.tile(fcol, (n, 1))
        flin = np.tile(flin, (1, m))
        inside=np.logical_and(flin, fcol)
        print("Bbox crosses the International Date Line.")

    ii, jj = np.meshgrid(range(m), range(n))
    iiin, jjin = ii[inside], jj[inside]
    i0, i1, j0, j1 = min(iiin), max(iiin), min(jjin), max(jjin)

    SPLIT_BBOX=(i1-i0)==(m-1) # Test whether the wanted bbox crosses edges of the longitude array.

    # If wanted bbox crosses edges of the longitude array, return indices for the two boxes separately.
    if SPLIT_BBOX:
        Iiin = np.unique(iiin)
        ib0 = np.diff(Iiin).argmax()  # Find edge of the inner side of the left bbox.
        ib1 = ib0 + 1                 # Find edge of the inner side of the right bbox.
        Il, Ir = Iiin[ib0], Iiin[ib1] # Indices of the columns that bound the inner side of the two bboxes.
        print("Bbox crosses edges of the longitude array. Returning two sets of indices.")
        return (i0, Il, j0, j1), (Ir, i1, j0, j1)
    else:
        return i0, i1, j0, j1

def get_arrdepth(arr):
    """
    USAGE
    -----
    arr_depths = get_arrdepth(arr)

    Determine number of nested levels in each
    element of an array of arrays of arrays...
    (or other array-like objects).
    """
    arr = np.array(arr) # Make sure first level is an array.

    all_nlevs = []
    for i in range(arr.size):
        nlev=0
        wrk_arr = arr[i]
        while np.size(wrk_arr)>0:
            try:
                wrk_arr = np.array(wrk_arr[i])
            except Exception:
                all_nlevs.append(nlev)
                nlev=0
                break
            nlev+=1

    return np.array(all_nlevs)

def fpointsbox(x, y, fig, ax, nboxes=1, plot=True):
    """
    USAGE
    -----
    fpts = fpointsbox(x, y, fig, ax, nboxes=1, plot=True)

    Find points in a rectangle made with 2 ginput points.
    """
    fpts = np.array([])
    for n in range(nboxes):
        box = np.array(fig.ginput(n=2, timeout=0))
        xb, yb = box[:,0], box[:,1]
        xl, xr, yd, yu = xb.min(), xb.max(), yb.min(), yb.max()
        xbox = np.array([xl, xr, xr, xl, xl])
        ybox = np.array([yd, yd, yu, yu, yd])
        fxbox, fybox = np.logical_and(x>xl, x<xr), np.logical_and(y>yd, y<yu)
        fptsi = np.logical_and(fxbox, fybox)
        fpts = np.append(fpts, fptsi)
        if plot:
            ax.plot(xbox, ybox, 'r', linestyle='solid', marker='o', ms=4)
            ax.plot(x[fptsi], y[fptsi], 'r', linestyle='none', marker='+', ms=5)
            fig.show()
        else:
            fig.close()

    return fpts

def near(x, x0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear = near(x, x0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in array 'x'
    that are closest to a specified 'x0' point.
    If 'return_index' is True (defauts to False),
    then the indices of the closest points are
    returned. The indices are ordered in order of
    closeness.
    """
    x = list(x)
    xnear = []
    xidxs = []
    for n in range(npts):
        idx = np.nanargmin(np.abs(np.array(x)-x0))
        xnear.append(x.pop(idx))
        if return_index:
            xidxs.append(idx)
    if return_index: # Sort indices according to the proximity of wanted points.
        xidxs = [xidxs[i] for i in np.argsort(xnear).tolist()]
    xnear.sort()

    if npts==1:
        xnear = xnear[0]
        if return_index:
            xidxs = xidxs[0]
    else:
        xnear = np.array(xnear)

    if return_index:
        return xidxs
    else:
        return xnear

def near2(x, y, x0, y0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear, ynear = near2(x, y, x0, y0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in arrays 'x' and 'y'
    that are closest to a specified '(x0, y0)' point. If
    'return_index' is True (defauts to False), then the
    indices of the closest point(s) are returned.

    Example
    -------
    >>> x = np.arange(0., 100., 0.25)
    >>> y = np.arange(0., 100., 0.25)
    >>> x, y = np.meshgrid(x, y)
    >>> x0, y0 = 44.1, 30.9
    >>> xn, yn = near2(x, y, x0, y0, npts=1)
    >>> print("(x0, y0) = (%f, %f)"%(x0, y0))
    >>> print("(xn, yn) = (%f, %f)"%(xn, yn))
    """
    x, y = map(np.array, (x, y))
    shp = x.shape

    xynear = []
    xyidxs = []
    dx = x - x0
    dy = y - y0
    dr = dx**2 + dy**2
    for n in range(npts):
        xyidx = np.unravel_index(np.nanargmin(dr), dims=shp)
        if return_index:
            xyidxs.append(xyidx)
        xyn = (x[xyidx], y[xyidx])
        xynear.append(xyn)
        dr[xyidx] = np.nan

    if npts==1:
        xynear = xynear[0]
        if return_index:
            xyidxs = xyidxs[0]

    if return_index:
        return xyidxs
    else:
        return xynear

def mnear(x, y, x0, y0):
	"""
	USAGE
	-----
	xmin,ymin = mnear(x, y, x0, y0)

	Finds the the point in a (lons,lats) line
	that is closest to a specified (lon0,lat0) point.
	"""
	x,y,x0,y0 = map(np.asanyarray, (x,y,x0,y0))
	point = (x0,y0)

	d = np.array([])
	for n in range(x.size):
		xn,yn = x[n],y[n]
		dn = distance((xn,x0),(yn,y0)) # Calculate distance point-wise.
		d = np.append(d,dn)

	idx = d.argmin()

	return x[idx],y[idx]

def refine(line, nref=100, close=True):
	"""
	USAGE
	-----
	ref_line = refine(line, nref=100, close=True)

	Given a 1-D sequence of points 'line', returns a
	new sequence 'ref_line', which is built by linearly
	interpolating 'nref' points between each pair of
	subsequent points in the original line.

	If 'close' is True (default), the first value of
	the original line is repeated at the end of the
	refined line, as in a closed polygon.
	"""
	line = np.squeeze(np.asanyarray(line))

	if close:
		line = np.append(line,line[0])

	ref_line = np.array([])
	for n in range(line.shape[0]-1):
		xi, xf = line[n], line[n+1]
		xref = np.linspace(xi,xf,nref)
		ref_line = np.append(ref_line, xref)

	return ref_line

def point_in_poly(x,y,poly):
	"""
	USAGE
	-----
	isinside = point_in_poly(x,y,poly)

	Determine if a point is inside a given polygon or not
	Polygon is a list of (x,y) pairs. This fuction
	returns True or False.  The algorithm is called
	'Ray Casting Method'.

	Source: http://pseentertainmentcorp.com/smf/index.php?topic=545.0
	"""
	n = len(poly)
	inside = False

	p1x,p1y = poly[0]
	for i in range(n+1):
		p2x,p2y = poly[i % n]
		if y > min(p1y,p2y):
			if y <= max(p1y,p2y):
				if x <= max(p1x,p2x):
					if p1y != p2y:
						xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or x <= xinters:
						inside = not inside
		p1x,p1y = p2x,p2y

	return inside

def get_mask_from_poly(xp, yp, poly, verbose=False):
	"""
	USAGE
	-----

	mask = get_mask_from_poly(xp, yp, poly, verbose=False)

	Given two arrays 'xp' and 'yp' of (x,y) coordinates (generated by meshgrid)
	and a polygon defined by an array of (x,y) coordinates 'poly', with
	shape = (n,2), return a boolean array 'mask', where points that lie inside
	'poly' are set to 'True'.
	"""
	print('Building the polygon mask...')
	jmax, imax = xp.shape
	mask = np.zeros((jmax,imax))
	for j in range(jmax):
		if verbose:
			print("Row %s of %s"%(j+1,jmax))
		for i in range(imax):
			px, py = xp[j,i], yp[j,i]
			# Test if this point is within the polygon.
			mask[j,i] = point_in_poly(px, py, poly)

	return mask

def sphericalpolygon_area(lons, lats, R=6371000.):
	"""
	USAGE
	-----
	area = sphericalpolygon_area(lons, lats, R=6371000.)

	Calculates the area of a polygon on the surface of a sphere of
	radius R using Girard's Theorem, which states that the area of
	a polygon of great circles is R**2 times the sum of the angles
	between the polygons minus (N-2)*pi, where N is number of corners.
	R = 6371000 m (6371 km, default) is a typical value for the mean
	radius of the Earth.

	Source: http://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python
	"""
	lons, lats = map(np.asanyarray, (lons, lats))
	N = lons.size

	angles = np.empty(N)
	for i in range(N):

	    phiB1, phiA, phiB2 = np.roll(lats, i)[:3]
	    LB1, LA, LB2 = np.roll(lons, i)[:3]

	    # calculate angle with north (eastward)
	    beta1 = greatCircleBearing(LA, phiA, LB1, phiB1)
	    beta2 = greatCircleBearing(LA, phiA, LB2, phiB2)

	    # calculate angle between the polygons and add to angle array
	    angles[i] = np.arccos(np.cos(-beta1)*np.cos(-beta2) + np.sin(-beta1)*np.sin(-beta2))

	return (np.sum(angles) - (N-2)*np.pi)*R**2

def greatCircleBearing(lon1, lat1, lon2, lat2):
	"""
	USAGE
	-----
	angle = greatCircleBearing(lon1, lat1, lon2, lat2)

	Calculates the angle (positive eastward) a
	great circle passing through points (lon1,lat1)
	and (lon2,lat2) makes with true nirth.

	Source: http://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python
	"""
	lon1, lat1, lon2, lat2 = map(np.asanyarray, (lon1, lat1, lon2, lat2))
	dLong = lon1 - lon2
	d2r = np.pi/180.

	s = np.cos(d2r*lat2)*np.sin(d2r*dLong)
	c = np.cos(d2r*lat1)*np.sin(d2r*lat2) - np.sin(lat1*d2r)*np.cos(d2r*lat2)*np.cos(d2r*dLong)

	return np.arctan2(s, c)

def weim(x, N, kind='hann', badflag=-9999, beta=14):
	"""
	Usage
	-----
	xs = weim(x, N, kind='hann', badflag=-9999, beta=14)

	Description
	-----------
	Calculates the smoothed array 'xs' from the original array 'x' using the specified
	window of type 'kind' and size 'N'. 'N' must be an odd number.

	Parameters
	----------
	x       : 1D array
	        Array to be smoothed.

	N       : integer
	        Window size. Must be odd.

	kind    : string, optional
	        One of the window types available in the numpy module:

	        hann (default) : Gaussian-like. The weight decreases toward the ends. Its end-points are zeroed.
	        hamming        : Similar to the hann window. Its end-points are not zeroed, therefore it is
	                         discontinuous at the edges, and may produce undesired artifacts.
	        blackman       : Similar to the hann and hamming windows, with sharper ends.
	        bartlett       : Triangular-like. Its end-points are zeroed.
	        kaiser         : Flexible shape. Takes the optional parameter "beta" as a shape parameter.
	                         For beta=0, the window is rectangular. As beta increases, the window gets narrower.

	        Refer to the numpy functions for details about each window type.

	badflag : float, optional
	        The bad data flag. Elements of the input array 'A' holding this value are ignored.

	beta    : float, optional
	        Shape parameter for the kaiser window. For windows other than the kaiser window,
	        this parameter does nothing.

	Returns
	-------
	xs      : 1D array
	        The smoothed array.

	---------------------------------------
	André Palóczy Filho (paloczy@gmail.com)
	June 2012
	==============================================================================================================
	"""
	###########################################
	### Checking window type and dimensions ###
	###########################################
	kinds = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']
	if ( kind not in kinds ):
		raise ValueError('Invalid window type requested: %s'%kind)

	if np.mod(N,2) == 0:
		raise ValueError('Window size must be odd')

	###########################
	### Creating the window ###
	###########################
	if ( kind == 'kaiser' ): # If the window kind is kaiser (beta is required).
		wstr = 'np.kaiser(N, beta)'
	else: # If the window kind is hann, hamming, blackman or bartlett (beta is not required).
		if kind == 'hann':
			kind = 'hanning'

	wstr = 'np.' + kind + '(N)'
	w = eval(wstr)

	x = np.asarray(x).flatten()
	Fnan = np.isnan(x).flatten()

	ln = (N-1)/2
	lx = x.size
	lf = lx - ln
	xs = np.nan*np.ones(lx)

	# Eliminating bad data from mean computation.
	fbad=x==badflag
	x[fbad] = np.nan

	for i in range(lx):
		if i <= ln:
			xx = x[:ln+i+1]
			ww = w[ln-i:]
		elif i >= lf:
			xx = x[i-ln:]
			ww = w[:lf-i-1]
		else:
			xx = x[i-ln:i+ln+1]
			ww = w.copy()

		f = ~np.isnan(xx) # Counting only NON-NaNs, both in the input array and in the window points.
		xx = xx[f]
		ww = ww[f]

		if f.sum() == 0: # Thou shalt not divide by zero.
			xs[i] = x[i]
		else:
			xs[i] = np.sum(xx*ww)/np.sum(ww)

		xs[Fnan] = np.nan # Assigning NaN to the positions holding NaNs in the input array.

	return xs

def smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14):
	"""
	Usage
	-----
	As = smoo2(A, hei, wid, kind='hann', badflag=-9999, beta=14)

	Description
	-----------
	Calculates the smoothed array 'As' from the original array 'A' using the specified
	window of type 'kind' and shape ('hei','wid').

	Parameters
	----------
	A       : 2D array
	        Array to be smoothed.

	hei     : integer
	        Window height. Must be odd and greater than or equal to 3.

	wid     : integer
	        Window width. Must be odd and greater than or equal to 3.

	kind    : string, optional
	        One of the window types available in the numpy module:

	        hann (default) : Gaussian-like. The weight decreases toward the ends. Its end-points are zeroed.
	        hamming        : Similar to the hann window. Its end-points are not zeroed, therefore it is
	                         discontinuous at the edges, and may produce undesired artifacts.
	        blackman       : Similar to the hann and hamming windows, with sharper ends.
	        bartlett       : Triangular-like. Its end-points are zeroed.
	        kaiser         : Flexible shape. Takes the optional parameter "beta" as a shape parameter.
	                         For beta=0, the window is rectangular. As beta increases, the window gets narrower.

	        Refer to the numpy functions for details about each window type.

	badflag : float, optional
	        The bad data flag. Elements of the input array 'A' holding this value are ignored.

	beta    : float, optional
	        Shape parameter for the kaiser window. For windows other than the kaiser window,
	        this parameter does nothing.

	Returns
	-------
	As      : 2D array
	        The smoothed array.

	---------------------------------------
	André Palóczy Filho (paloczy@gmail.com)
	April 2012
	==============================================================================================================
	"""
	###########################################
	### Checking window type and dimensions ###
	###########################################
	kinds = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']
	if ( kind not in kinds ):
		raise ValueError('Invalid window type requested: %s'%kind)

	if ( np.mod(hei,2) == 0 ) or ( np.mod(wid,2)  == 0 ):
		raise ValueError('Window dimensions must be odd')

	if (hei <= 1) or (wid <= 1):
		raise ValueError('Window shape must be (3,3) or greater')

	##############################
	### Creating the 2D window ###
	##############################
	if ( kind == 'kaiser' ): # If the window kind is kaiser (beta is required).
		wstr = 'np.outer(np.kaiser(hei, beta), np.kaiser(wid, beta))'
	else: # If the window kind is hann, hamming, blackman or bartlett (beta is not required).
		if kind == 'hann':
			kind = 'hanning'

		# computing outer product to make a 2D window out of the original 1d windows.
		wstr = 'np.outer(np.' + kind + '(hei), np.' + kind + '(wid))'
		wdw = eval(wstr)

	A = np.asanyarray(A)
	Fnan = np.isnan(A)
	imax, jmax = A.shape
	As = np.nan*np.ones( (imax, jmax) )

	for i in range(imax):
		for j in range(jmax):
			### Default window parameters.
			wupp = 0
			wlow = hei
			wlef = 0
			wrig = wid
			lh = np.floor(hei/2)
			lw = np.floor(wid/2)

			### Default array ranges (functions of the i,j indices).
			upp = i-lh
			low = i+lh+1
			lef = j-lw
			rig = j+lw+1

			##################################################
			### Tiling window and input array at the edges ###
			##################################################
			# Upper edge.
			if upp < 0:
				wupp = wupp-upp
				upp = 0

			# Left edge.
			if lef < 0:
				wlef = wlef-lef
				lef = 0

			# Bottom edge.
			if low > imax:
				ex = low-imax
				wlow = wlow-ex
				low = imax

			# Right edge.
			if rig > jmax:
				ex = rig-jmax
				wrig = wrig-ex
				rig = jmax

			###############################################
			### Computing smoothed value at point (i,j) ###
			###############################################
			Ac = A[upp:low, lef:rig]
			wdwc = wdw[wupp:wlow, wlef:wrig]
			fnan = np.isnan(Ac)
			Ac[fnan] = 0; wdwc[fnan] = 0 # Eliminating NaNs from mean computation.
			fbad = Ac==badflag
			wdwc[fbad] = 0               # Eliminating bad data from mean computation.
			a = Ac * wdwc
			As[i,j] = a.sum() / wdwc.sum()

	As[Fnan] = np.nan # Assigning NaN to the positions holding NaNs in the input array.

	return As

def denan(arr):
	"""
	USAGE
	-----
	denaned_arr = denan(arr)

	Remove the NaNs from an array.
	"""
	f = np.isnan(arr)
	return arr[~f]

def standardize(series):
	"""
	USAGE
	-----
	series2 = standardize(series)

	Standardizes a series by subtracting its mean value
	and dividing by its standard deviation. The result is
	a dimensionless series. Inputs can be of type
	"np.array", or "Pandas.Series"/"Pandas.TimeSeries".
	"""
	Mean, Std = series.mean(), series.std()
	return (series - Mean)/Std

def linear_trend(series, return_line=True):
	"""
	USAGE
	-----
	line = linear_trend(series, return_line=True)

	OR

	b, a, x = linear_trend(series, return_line=False)

	Returns the linear fit (line = b*x + a) associated
	with the 'series' array.

	Adapted from pylab.detrend_linear.
	"""
	series = np.asanyarray(series)
	x = np.arange(series.size, dtype=np.float_)

	C = np.cov(x, series, bias=1) # Covariance matrix.
	b = C[0, 1]/C[0, 0] # Angular coefficient.

	a = series.mean() - b*x.mean() # Linear coefficient.
	line = b*x + a

	if return_line:
		return line
	else:
		return b, a, x

def thomas(A, b):
    """
    USAGE
    -----
    x = thomas(A,b)

    Solve Ax = b (where A is a tridiagonal matrix)
    using the Thomas Algorithm.

    References
    ----------
    For a step-by-step derivation of the algorithm, see
    e.g., http://www3.ul.ie/wlee/ms6021_thomas.pdf
    """
    # Step 1: Sweep rows from top to bottom,
    # calculating gammas and rhos along the way.
    N = b.size
    gam = [float(A[0,1]/A[0,0])]
    rho = [float(b[0]/A[0,0])]
    for i in range(0, N):
        rho.append(float((b[i] - A[i,i-1]*rho[-1])/(A[i,i] - A[i,i-1]*gam[-1])))
        if i<N-1: # No gamma in the last row.
            gam.append(float(A[i,i+1]/(A[i,i] - A[i,i-1]*gam[-1])))

    # Step 2: Substitute solutions for unknowns
    # starting from the bottom row all the way up.
    x = [] # Vector of unknowns.
    x.append(rho.pop()) # Last row is already solved.
    for i in range(N-2, -1, -1):
        x.append(float(rho.pop() - gam.pop()*x[-1]))

    x.reverse()
    return np.array(x)

def topo_slope(lon, lat, h):
	"""
	USAGE
	-----
	lons, lats, slope = topo_slope(lon, lat, h)

	Calculates bottom slope for a topography fields 'h' at
	coordinates ('lon', 'lat') using first-order finite differences.
	The output arrays have shape (M-1,L-1), where M,L = h.shape().
	"""
	lon,lat,h = map(np.asanyarray, (lon,lat,h))
	deg2m = 1852.*60.    # m/deg.
	deg2rad = np.pi/180. # rad/deg.

	x = lon*deg2m*np.cos(lat*deg2rad)
	y = lat*deg2m

	# First-order differences, accurate to O(dx) and O(dy),
	# respectively.
	sx = (h[:,1:] - h[:,:-1]) / (x[:,1:] - x[:,:-1])
	sy = (h[1:,:] - h[:-1,:]) / (y[1:,:] - y[:-1,:])

	# Finding the values of the derivatives sx and sy
	# at the same location in physical space.
	sx = 0.5*(sx[1:,:]+sx[:-1,:])
	sy = 0.5*(sy[:,1:]+sy[:,:-1])

	# Calculating the bottom slope.
	slope = np.sqrt(sx**2 + sy**2)

	# Finding the lon,lat coordinates of the
	# values of the derivatives sx and sy.
	lons = 0.5*(lon[1:,:]+lon[:-1,:])
	lats = 0.5*(lat[1:,:]+lat[:-1,:])
	lons = 0.5*(lons[:,1:]+lons[:,:-1])
	lats = 0.5*(lats[:,1:]+lats[:,:-1])

	return lons, lats, slope

def curvature_geometric(x, y):
	"""
	USAGE
	-----
	k = curvature_geometric(x, y)

	Estimates the curvature k of a 2D curve (x,y) using a geometric method.

	If your curve is given by two arrays, x and y, you can
	approximate its curvature at each point by the reciprocal of the
	radius of a circumscribing triangle with that point, the preceding
	point, and the succeeding point as vertices. The radius of such a
	triangle is one fourth the product of the three sides divided by its
	area.

	The curvature will be positive for curvature to the left and
	negative for curvature to the right as you advance along the curve.

	Note that if your data are too closely spaced together or subject
	to substantial noise errors, this formula will not be very accurate.

	Author: Roger Stafford
	Source: http://www.mathworks.com/matlabcentral/newsreader/view_thread/125637
	Translated to Python by André Palóczy, January 19, 2015.
	"""
	x,y = map(np.asanyarray, (x,y))

	x1 = x[:-2]; x2 = x[1:-1]; x3 = x[2:]
	y1 = y[:-2]; y2 = y[1:-1]; y3 = y[2:]
	## a, b, and c are the three sides of the triangle.
	a = np.sqrt((x3-x2)**2 + (y3-y2)**2)
	b = np.sqrt((x1-x3)**2 + (y1-y3)**2)
	c = np.sqrt((x2-x1)**2 + (y2-y1)**2)
	## A is the area of the triangle.
	A = 0.5*(x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)
	## The reciprocal of the circumscribed radius, i.e., the curvature.
	k = 4.0*A/(a*b*c)

	return np.squeeze(k)

def get_isobath(lon, lat, topo, iso):
	"""
	USAGE
	-----
	lon_isob, lat_isob = get_isobath(lon_topo, lat_topo, topo, iso)

	Retrieves the 'lon_isob','lat_isob' coordinates of a wanted 'iso'
	isobath from a topography array 'topo', with 'lon_topo','lat_topo'
	coordinates.
	"""
	lon, lat, topo = map(np.asanyarray, (lon, lat, topo))

	plt.ioff()
	fig, ax = plt.subplots()
	cs = ax.contour(lon, lat, topo, [iso])
	coll = cs.collections[0]
	## Test all lines to find thel ongest one.
	## This is assumed to be the wanted isobath.
	ncoll = len(coll.get_paths())
	siz = np.array([])
	for n in range(ncoll):
		path = coll.get_paths()[n]
		siz = np.append(siz, path.vertices.shape[0])

	f = siz.argmax()
	xiso = coll.get_paths()[f].vertices[:, 0]
	yiso = coll.get_paths()[f].vertices[:, 1]
	del(fig, ax, cs)
	plt.ion()

	return xiso, yiso

def angle_isobath(lon, lat, h, isobath=100., smooth_isobath=True, window_length=21, plot_map=False):
    """
    USAGE
    -----
    lon_isob, lat_isob, angle = angle_isobath(lon, lat, h, isobath=100., smooth_isobath=True, window_length=21, plot_map=False)

    Returns the coordinates ('lon_isob', 'lat_isob') and the angle an isobath (defaults to the 100 m isobath)
    makes with the zonal direction for a topography array 'h' at coordinates ('lon', 'lat'). If 'smooth_isobath'==True,
    smooths the isobath with a Hanning window that is 'window_length' points wide. If 'plot_map'==True, plots a map showing
    the isobath (and its soothed version if smooth_isobath==True).
    """
    lon, lat, h = map(np.asanyarray, (lon, lat, h))
    R = 6371000.0 # Mean radius of the earth in meters (6371 km), from gsw.constants.earth_radius.
    deg2rad = np.pi/180. # [rad/deg]

    ## Extract isobath coordinates
    xiso, yiso = get_isobath(lon, lat, h, isobath)

    ## Smooth the isobath.
    if smooth_isobath:
        xisosm = weim(xiso, window_length, kind='hann')
        yisosm = weim(yiso, window_length, kind='hann')

    ## From the coordinates of the isobath, find the angle it forms with the
    ## zonal axis, using points k+1 and k.
    shth = yiso.size-1
    theta = np.zeros(shth)
    for k in range(shth):
        dyk = R*(yiso[k+1]-yiso[k])
        dxk = R*(xiso[k+1]-xiso[k])*np.cos(yiso[k]*deg2rad)
        theta[k] = np.arctan2(dyk,dxk)

    if smooth_isobath:
        thetasm = np.zeros(shth)
        for k in range(shth):
            dyk = R*(yisosm[k+1]-yisosm[k])
            dxk = R*(xisosm[k+1]-xisosm[k])*np.cos(yisosm[k]*deg2rad)
            thetasm[k] = np.arctan2(dyk,dxk)

    # if np.isnan(theta).sum()==0.:
    xisom = 0.5*(xiso[1:]+xiso[:-1])
    yisom = 0.5*(yiso[1:]+yiso[:-1])

    if smooth_isobath:
        xisomsm = 0.5*(xisosm[1:]+xisosm[:-1])
        yisomsm = 0.5*(yisosm[1:]+yisosm[:-1])

    ## Plots map showing the extracted isobath (and its smoothed version if smooth_isobath=True).
    if plot_map:
        fig, ax = plt.subplots()
        m = bb_map([lon.min(), lon.max()], [lat.min(), lat.max()], projection='cyl', resolution='h', ax=ax)
        m.plot(xisom, yisom, color='b', linestyle='-', zorder=3, latlon=True)
        if smooth_isobath:
            m.plot(xisomsm, yisomsm, color='r', linestyle='-', zorder=3, latlon=True)
        raw_input("Press any key to continue.")
        plt.close()

    if smooth_isobath:
        return xisomsm, yisomsm, thetasm
    else:
        return xisom, yisom, theta

def isopyc_depth(z, dens0, isopyc=1027.75, dzref=1.):
    """
    USAGE
    -----
    hisopyc = isopyc_depth(z, dens0, isopyc=1027.75)

    Calculates the spatial distribution of the depth of a specified isopycnal 'isopyc'
    (defaults to 1027.75 kg/m3) from a 3D density array rho0 (in kg/m3) with shape
    (nz,ny,nx) and a 1D depth array 'z' (in m) with shape (nz).

    'dzref' is the desired resolution for the refined depth array (defaults to 1 m) which
    is generated for calculating the depth of the isopycnal. The smaller 'dzref', the smoother
    the resolution of the returned isopycnal depth array 'hisopyc'.
    """
    z, dens0 = map(np.asanyarray, (z, dens0))
    ny, nx = dens0.shape[1:]
    zref = np.arange(z.min(), z.max(), dzref)

    if np.ma.isMaskedArray(dens0):
        dens0 = np.ma.filled(dens0, np.nan)

    hisopyc = np.nan*np.ones((ny,nx))
    for j in range(ny):
        for i in range(nx):
            dens0ij = dens0[:,j,i]
            if np.logical_or(np.logical_or(isopyc<np.nanmin(dens0ij), np.nanmax(dens0ij)<isopyc), np.isnan(dens0ij).all()):
                continue
            else:
                dens0ref = np.interp(zref, z, dens0ij) # Refined density profile.
                dens0refn = near(dens0ref, isopyc)
                fz=dens0ref==dens0refn
                try:
                    hisopyc[j,i] = zref[fz]
                except ValueError:
                    print("Warning: More than 1 (%d) nearest depths found. Using the median of the depths for point (j=%d,i=%d)."%(fz.sum(), j, i))
                    hisopyc[j,i] = np.nanmedian(zref[fz])

    return hisopyc

def whiten_zero(x, y, z, ax, cs, n=1, cmap=plt.cm.RdBu_r, zorder=9):
	"""
	USAGE
	-----
	whiten_zero(x, y, z, ax, cs, n=1, cmap=plt.cm.RdBu_r, zorder=9)

	Changes to white the color of the 'n' (defaults to 1)
	neighboring patches about the zero contour created
	by a command like 'cs = ax.contourf(x, y, z)'.
	"""
	x, y, z = map(np.asanyarray, (x,y,z))
	white = (1.,1.,1.)
	cslevs = cs.levels
	assert 0. in cslevs
	f0=np.where(cslevs==0.)[0][0]
	f0m, f0p = f0-n, f0+n
	c0m, c0p = cslevs[f0m], cslevs[f0p]
	ax.contourf(x, y, z, levels=[c0m, c0p], linestyles='none', colors=[white, white], cmap=None, zorder=zorder)

def wind2stress(u, v, formula='large_pond1981-modified'):
	"""
	USAGE
	-----
	taux,tauy = wind2stress(u, v, formula='mellor2004')

	Converts u,v wind vector components to taux,tauy
	wind stress vector components.
	"""
	rho_air = 1.226            # kg/m3
	mag = np.sqrt(u**2+v**2)   # m/s
	Cd = np.zeros( mag.shape ) # Drag coefficient.

	if formula=='large_pond1981-modified':
		# Large and Pond (1981) formula
		# modified for light winds, as
		# in Trenberth et al. (1990).
		f=mag<=1.
		Cd[f] = 2.18e-3
		f=np.logical_and(mag>1.,mag<3.)
		Cd[f] = (0.62+1.56/mag[f])*1e-3
		f=np.logical_and(mag>=3.,mag<10.)
		Cd[f] = 1.14e-3
		f=mag>=10.
		Cd[f] = (0.49 + 0.065*mag[f])*1e-3
	elif formula=='mellor2004':
		Cd = 7.5e-4 + 6.7e-5*mag
	else:
		np.disp('Unknown formula for Cd.')
		pass

	# Computing wind stress [N/m2]
	taux = rho_air*Cd*mag*u
	tauy = rho_air*Cd*mag*v

	return taux,tauy

def gen_dates(start, end, dt='day', input_datetime=False):
	"""
	Returns a list of datetimes within the date range
	from `start` to `end`, at a `dt` time interval.

	`dt` can be 'second', 'minute', 'hour', 'day', 'week',
	'month' or 'year'.

	If `input_datetime` is False (default), `start` and `end`
	must be a date in string form. If `input_datetime` is True,
	`start` and `end` must be datetime objects.

	Note
	----
	Modified from original function
	by Filipe Fernandes (ocefpaf@gmail.com).

	Example
	-------
	>>> from ap_tools.utils import gen_dates
	>>> from datetime import datetime
	>>> start = '1989-08-19'
	>>> end = datetime.utcnow().strftime("%Y-%m-%d")
	>>> gen_dates(start, end, dt='day')
	"""
	DT = dict(second=rrule.SECONDLY,
		      minute=rrule.MINUTELY,
		      hour=rrule.HOURLY,
		      day=rrule.DAILY,
		      week=rrule.WEEKLY,
		      month=rrule.MONTHLY,
		      year=rrule.YEARLY)

	dt = DT[dt]

	if input_datetime: # Input are datetime objects. No parsing needed.
		dates = rrule.rrule(dt, dtstart=start, until=end)
	else:              # Input in string form, parse into datetime objects.
		dates = rrule.rrule(dt, dtstart=parser.parse(start), until=parser.parse(end))
	return list(dates)

def fmt_isobath(cs, fontsize=8, fmt='%g', inline=True, inline_spacing=7, manual=True):
	"""
	Formats the labels of isobath contours. `manual` is set to `True` by default,
	but can be `False`, or a tuple/list of tuples with the coordinates of the labels.
	All options are passed to plt.clabel().
	"""
	isobstrH = plt.clabel(cs, fontsize=fontsize, fmt=fmt, inline=inline, inline_spacing=inline_spacing, manual=manual)
	for ih in range(0, len(isobstrH)): # Appends 'm' for meters at the end of the label.
		isobstrh = isobstrH[ih]
		isobstr = isobstrh.get_text()
		isobstr = isobstr.replace('-','') + ' m'
		isobstrh.set_text(isobstr)

def float2latex(f, ndigits=1):
	"""
	USAGE
	-----
	texstr = float2latex(f, ndigits=1)

	Converts a float input into a latex-formatted
	string with 'ndigits' (defaults to 1).

	Adapted from:
	http://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
	"""
	float_str = "{0:.%se}"%ndigits
	float_str = float_str.format(f)
	base, exponent = float_str.split("e")
	return "${0} \times 10^{{{1}}}$".format(base, int(exponent))

def mat2npz(matname):
	"""
	USAGE
	-----
	mat2npz(matname)

	Extract variables stored in a .mat file,
	and saves them in a .npz file.
	"""
	d = loadmat(matname)
	_ = d.pop('__header__')
	_ = d.pop('__globals__')
	_ = d.pop('__version__')
	npzname = matname[:-4] + '.npz'
	np.savez(npzname,**d)
	return None

def bb_map(lons, lats, ax, projection='merc', resolution='i', drawparallels=True, drawmeridians=True):
	"""
	USAGE
	-----
	m = bb_map(lons, lats, **kwargs)

	Returns a Basemap instance with lon,lat bounding limits
	inferred from the input arrays `lons`,`lats`.
	Coastlines, countries, states, parallels and meridians
	are drawn, and continents are filled.
	"""
	lons,lats = map(np.asanyarray, (lons,lats))
	lonmin,lonmax = lons.min(),lons.max()
	latmin,latmax = lats.min(),lats.max()

	m = Basemap(llcrnrlon=lonmin,
				urcrnrlon=lonmax,
				llcrnrlat=latmin,
				urcrnrlat=latmax,
				projection=projection,
				resolution=resolution,
				ax=ax)

	plt.ioff() # Avoid showing the figure.
	m.fillcontinents(color='0.9', zorder=9)
	m.drawcoastlines(zorder=10)
	m.drawstates(zorder=10)
	m.drawcountries(linewidth=2.0, zorder=10)
	m.drawmapboundary(zorder=9999)
	if drawmeridians:
		m.drawmeridians(np.arange(np.floor(lonmin), np.ceil(lonmax), 1), linewidth=0.15, labels=[1, 0, 1, 0], zorder=12)
	if drawparallels:
		m.drawparallels(np.arange(np.floor(latmin), np.ceil(latmax), 1), linewidth=0.15, labels=[1, 0, 0, 0], zorder=12)
	plt.ion()
	return m

def dots_dualcolor(x, y, z, thresh=20., color_low='b', color_high='r', marker='o', markersize=5):
	"""
	USAGE
	-----
    dots_dualcolor(x, y, z, thresh=20., color_low='b', color_high='r')

	Plots dots colored with a dual-color criterion,
	separated by a threshold value.
	"""
	ax = plt.gca()
	# Below-threshold dots.
	f=z<=thresh
	ax.plot(x[f], y[f], lw=0, marker=marker, ms=markersize, mfc=color_low, mec=color_low)
	# Above-threshold dots.
	f=z>thresh
	ax.plot(x[f], y[f], lw=0, marker=marker, ms=markersize, mfc=color_high, mec=color_high)

if __name__=='__main__':
  import doctest
  doctest.testmod()
