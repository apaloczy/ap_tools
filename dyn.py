# Description: Functions to calculate dynamical quantities.
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['deg2m_dist',
           'divergence',
           'strain',
           'vorticity',
           'pgf']

import numpy as np
from gsw import grav

def deg2m_dist(lon, lat):
	"""
	USAGE
	-----
	dx, dy = deg2m_dist(lon, lat)

	Calculates zonal and meridional grid spacing 'dx' and 'dy' (in meters)
	from the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees), using centered
	(forward/backward) finite-differences for the interior (edge) points. Assumes
    a locally rectangular cartesian on the scales of 'dx' and 'dy'.
	"""
	lon, lat = map(np.array, (lon, lat))

	dlat, _ = np.gradient(lat)             # [deg]
	_, dlon = np.gradient(lon)             # [deg]
	deg2m = 111120.0                       # [m/deg]
	# Account for divergence of meridians in zonal distance.
	dx = dlon*deg2m*np.cos(lat*np.pi/180.) # [m]
	dy = dlat*deg2m                        # [m]

	return dx, dy

def divergence(lon, lat, u, v):
	"""
	USAGE
	-----
	div = divergence(lon, lat, u, v)

	Calculates horizontal divergence 'div' (du/dx + dv/dy, in 1/s) from
	the 'u' and 'v' velocity arrays (in m/s) specified in spherical
	coordinates by the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees).
	"""
	lon, lat, u, v = map(np.array, (lon, lat, u, v))

	dx, dy = deg2m_dist(lon, lat) # [m]
	_, dux = np.gradient(u)
	dvy, _ = np.gradient(v)

	dudx = dux/dx
	dvdy = dvy/dy
	div = dudx + dvdy # [1/s]

	return div

def strain(lon, lat, u, v):
    """
    USAGE
    -----
    alpha = strain(lon, lat, u, v)

    Calculates lateral rate of strain 'alpha' = sqrt[(du/dx - dv/dy)^2 + (du/dy + dv/dx)^2],
    in 1/s, from the 'u' and 'v' velocity arrays (in m/s) specified in spherical coordinates
    by the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees).
    """
    lon, lat, u, v = map(np.array, (lon, lat, u, v))

    dx, dy = deg2m_dist(lon, lat) # [m]
    duy, dux = np.gradient(u)
    dvy, dvx = np.gradient(v)

    dudx = dux/dx
    dvdy = dvy/dy
    dudy = duy/dy
    dvdx = dvx/dx
    alpha = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2) # [1/s]

    return alpha

def vorticity(x, y, u, v, coord_type='geographic'):
    """
    USAGE
    -----
    zeta = vorticity(x, y, u, v, coord_type='geographic')

    Calculates the vertical component 'zeta' (dv/dx - du/dy, in 1/s) of the
    relative vorticity vector from the 'u' and 'v' velocity arrays (in m/s)
    specified in spherical coordinates by the 'lon' and 'lat' 2D meshgrid-type
    arrays (in degrees).
    """
    x, y, u, v = map(np.array, (x, y, u, v))

    if coord_type=='geographic':
        dx, dy = deg2m_dist(lon, lat)
    elif coord_type=='cartesian':
        dy, _ = np.gradient(y)
        _, dx = np.gradient(x)
    elif coord_type=='dxdy':
        dx, dy = x, y
        pass

    duy, _ = np.gradient(u)
    _, dvx = np.gradient(v)

    dvdx = dvx/dx
    dudy = duy/dy
    vrt = dvdx - dudy # [1/s]

    return vrt

def pgf(z, y, x, eta, rho, pa=0., rho0=1025., geographic=True):
	"""
	USAGE
	-----
	Py, Px = pgf(z, y, x, eta, rho, pa=0., rho0=1025., geographic=True)

	Calculates total horizontal pressure gradient force per unit mass [m/s2]
	(barotropic + baroclinic + barometric components), i.e.,

	P(z,y,x) = -(1/rho0)*grad_H(pa) -g*grad_H(eta) + (g/rho0)*Integral{grad_H(rho)}.

	'rho0' is the reference potential density used in the Boussinesq Approximation
	(defaults to 1025. kg/m3), 'g' is the gravitational acceleration, 'pa(y,x)' is
	the atmospheric pressure in [N/m2] (defults to 0.), 'eta(y,x)' is the free surface elevation in [m],
	'rho(z,y,x)' is the potential density and 'grad_H' is the horizontal gradient operator.
	The Integral(rho) is calculated from z' = eta(x,y) down through z' = z.

	The coordinate arrays (z,y,x) are distances in the (vertical,meridional,zonal)
	directions. The vertical axis originates at the surface (z = 0), i.e.,
	rho[0,y,x] = rho_surface and
	rho[-1,y,x] = rho_bottom.

	If geographic==True (default), (y,x) are assumed to be
	(latitude,longitude) and are converted to meters before
	computing (dy,dx). If geographic==False, (y,x) are assumed to be in meters.
	"""
	z, y, x, eta, rho = map(np.array, (z, y, x, eta, rho))

	ny, nx = eta.shape                       # Shape of the (x,y,u,v) arrays.
	if z.ndim==1:
		z = np.expand_dims(z, 1)
		z = np.expand_dims(z, 1)
		z = np.tile(z, (1, ny, nx))

	## Calculating grid spacings.
	if geographic:
		dx, dy = deg2m_dist(lon, lat)
	else:
		dy, _ = np.gradient(y)
		_, dx = np.gradient(x)

	dz, _, _ = np.gradient(z)
	dz = np.abs(dz)

	## Get gravitational acceleration.
	if geographic:
		g = grav(y)
	else:
		g = 9.81

	## pa (x,y) derivatives.
	if pa:
		dpay, dpax = np.gradient(pa)
		dpady = dpay/dy
		dpadx = dpax/dx

	## eta (x,y) derivatives.
	detay, detax = np.gradient(eta)
	detady = detay/dy
	detadx = detax/dx

	## rho (x,y) derivatives.
	_, drhoy, drhox = np.gradient(rho)
	drhody = drhoy/dy
	drhodx = drhox/dx

	## Barometric pressure gradient force per unit mass.
	if pa==0.:
		PGF_bm_y, PGF_bm_x = np.zeros((ny, nx)), np.zeros((ny, nx))
	else:
		PGF_bm_y = -dpady/rho0
		PGF_bm_x = -dpadx/rho0

	## Barotropic pressure gradient force per unit mass.
	PGF_bt_y = -g*detady
	PGF_bt_x = -g*detadx

	## Vertical integration from z' = eta(x,y) through z' = z.
	Iy = np.cumsum(drhody*dz, axis=0)
	Ix = np.cumsum(drhodx*dz, axis=0)

	## Baroclinic pressure gradient force per unit mass.
	PGF_bc_y = +g*Iy/rho0
	PGF_bc_x = +g*Ix/rho0

	## Total pressure gradient force per unit mass.
	PGF_y = PGF_bm_y + PGF_bt_y + PGF_bc_y
	PGF_x = PGF_bm_x + PGF_bt_x + PGF_bc_x

	return PGF_y, PGF_x
