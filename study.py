# -*- coding: utf-8 -*-
#
# Description: Coursework-related scripts.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

__all__ = ['ekman','dynmodes']

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
try:
	from romslab import near
except:
	try:
		from romslab.romslab import near
	except:
		pass
try:
	from oceans.plotting import rstyle
except:
	pass

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

def ekman(r=4., plot=True, savefig=False):
	"""
	Plots the solution of the Ekman problem
	in a vertically finite ocean of depth H.
	Wind is purely meridional, f-plane, linearized
	steady-state linearized solutions. The wind is
	imposed impulsively over the initially at rest
	water column.
	"""
	tau0y = 0.08                          # Wind stress, meridional-only [Pa]
	rho = 1025.                           # Seawater density             [kg m^{-3}]
	A = 5e-2                              # Eddy viscosity coefficient   [m^{2} s^{-1}]
	theta0 = -30.                         # (Reference) latitude.                            [ ]
	omega = 2*np.pi/86400.                # Planetary angular frequency.                     [s^{-1}]
	f = 2*omega*np.sin(theta0*np.pi/180.) # Coriolis parameter.                              [s^{-1}]
	d = np.sqrt(2*A/np.abs(f))            # Ekman layer depth.                               [m]
	H = r*d                               # Bottom depth.                                    [m]
	st = "H, d, H/d = %s, %s, %s"%(str(H),str(d),str(r))
	np.disp(st)
	z = np.linspace(0, H, 1000.)

	# Solutions for u and v.
	C = tau0y/(rho*A)
	Denom = np.sinh(r)*np.sin(r) + np.cosh(r)*np.cos(r)
	u = C*np.sinh(z/d)*np.cos(z/d)/Denom
	v = C*np.cosh(z/d)*np.sin(z/d)/Denom

	# Veering angle at the surface.
	ang = np.arctan(v[-1]/u[-1])*180/np.pi
	st = "Veering angle at the surface: %s"%str(ang)
	np.disp(st)

	if plot:
		# u and v profiles.
		plt.close('all')
		fig,ax = plt.subplots()
		ax.hold(True)
		ax.plot(u,z,'k-',label='Zonal velocity, u$_{Ek}$')
		ax.plot(v,z,'k--',label='Meridional velocity, v$_{Ek}$')
		ax.set_xlabel('Velocity [cm s$^{-1}$]')
		ax.set_ylabel('Depth [m]')
		ax.legend(loc='best',fontsize=13)
		ax.hold(False)
		try:
			rstyle(ax)
		except:
			pass
		if savefig:
			plt.savefig('uv.png',bbox='tight')

		# Hodograph plot.
		fig,ax = plt.subplots()
		ax.hold(True)
		ax.plot(u,v,'k-',linewidth=1.5)
		# sx = ax.scatter(u,v,c=z,cmap=plt.cm.binary,linewidths=0.,marker='o')
		# cb = plt.colorbar(mappable=sx)
		# cb.set_label('Depth [m]')
		ax.set_xlabel('Zonal velocity [cm s$^{-1}$]')
		ax.set_ylabel('Meridional velocity [cm s$^{-1}$]')
		ax.hold(False)
		try:
			rstyle(ax)
		except:
			pass
		if savefig:
			plt.savefig('hodo.png',bbox='tight')

	return z,u,v

def dynmodes(n=6, lat0=5., plot=False, model='Fratantoni_etal1995'):
	"""
	Computes the discrete eigenmodes (dynamical modes)
	for a quasi-geostrophic ocean with n isopycnal layers.
	Rigid lids are placed at the surface and the bottom.

	Inputs:
	-------
	n:    Number of layers.
	lat0: Reference latitude.
	H:    List of rest depths of each layer.
    S:    List of potential density values for each layer.
	"""
	omega = 2*np.pi/86400.              # [rad s^{-1}]
	f0 = 2*omega*np.sin(lat0*np.pi/180) # [s^{-1}]
	f02 = f0**2                         # [s^{-2}]
	g = 9.81                            # [m s^{-1}]
	rho0 = 1027.                        # [kg m^{-3}]

	if model=='Fratantoni_etal1995':
		H = np.array([80.,170.,175.,250.,325.,3000.])
		S = np.array([24.97,26.30,26.83,27.12,27.32,27.77])
		tit = 'Modelo de seis camadas para a CNB de Fratantoni \textit{et al.} (1995)'
		figname = 'vmodes_fratantoni_etal1995'
	elif model=='Bub_Brown1996':
		H = np.array([150.,440.,240.,445.,225.,2500.])
		S = np.array([24.13,26.97,27.28,27.48,27.74,27.87])
		tit = 'Modelo de seis camadas para a CNB de Bub e Brown (1996)'
		figname = 'vmodes_bub_brown1996'

	# Normalized density jumps.
	E = (S[1:]-S[:-1])/rho0
	# Rigid lids at the surface and the bottom,
	# meaning infinite density jumps.
	E = np.hstack( (np.inf,E,np.inf) )

	# Building the tridiagonal matrix.
	A = np.zeros( (n,n) )
	for i in xrange(n):
		A[i,i] = -f02/(E[i+1]*g*H[i]) -f02/(E[i]*g*H[i]) # The main diagonal.
		if i>0:
			A[i,i-1] = f02/(E[i]*g*H[i])
		if i<(n-1):
			A[i,i+1] = f02/(E[i+1]*g*H[i])

	# get eigenvalues and convert them
	# to internal deformation radii
	lam,v = eig(A)
	lam = np.abs(lam)

	# Baroclinic def. radii in km:
	uno = np.ones( (lam.size,lam.size) )
	Rd = 1e-3*uno/np.sqrt(lam)
	Rd = np.unique(Rd)
	Rd = np.flipud(Rd)

	np.disp("Deformation radii [km]:")
	[np.disp(int(r)) for r in Rd]

	# orthonormalize eigenvectors, i.e.,
	# find the dynamical mode vertical structure functions.
	F = np.zeros( (n,n) )

	for i in xrange(n):
		mi = v[:,i] # The vertical structure of the i-th vertical mode.
		fac = np.sqrt(np.sum(H*mi*mi)/np.sum(H))
		F[:,i] = 1/fac*mi

	F=-F
	F[:,0] = np.abs(F[:,1])
	F = np.fliplr(F)

	Fi = np.vstack( (F[0,:],F) )
	Fi = np.flipud(Fi)
	for i in xrange(n-1):
		Fi[i,:] = F[i+1,:]
	Fi = np.flipud(Fi)

	# Plot the vertical modes.
	if plot:
		plt.close('all')
		kw = dict(fontsize=15, fontweight='black')
		fig,ax = plt.subplots()
		ax.hold(True)
		Hc = np.sum(H)
		z = np.flipud(np.linspace(-Hc,0,1000))
		Hp = np.append(0,H)
		Hp = -np.cumsum(Hp)

		# build modes for plotting purposes
		Fp = np.zeros( (z.size,n) )
		fo = 0

		for i in xrange(n):
			f1 = near(z,Hp[i])[0][0]
			for j in xrange(fo,f1):
				Fp[j,:] = F[i,:]
				fo=f1

		for i in xrange(n):
			l = 'Modo %s'%str(i)
			ax.plot(Fp[:,i], z, label=l)
		xl,xr = ax.set_xlim(-5,5)
		ax.hlines(Hp,xl,xr,linestyle='dashed')
		ax.hold(False)
		ax.set_title(tit, **kw)
		ax.set_xlabel('Autofunção [adimensional]', **kw)
		ax.set_ylabel('Profundidade [m]', **kw)
		try:
			rstyle(ax)
		except:
			pass
		ax.legend(loc='lower left', fontsize=20, fancybox=True, shadow=True)
		fmt='png'
		fig.savefig(figname+'.'+fmt, format=fmt, bbox='tight')
