# Description: Functions to fit statistical models to data series.
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com
# Description: Functions to fit statistical models to data series.

__all__ = ['sinfitn',
           'fourfilt']

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

def sinfitn(x, t, periods, constant=True, line=True, return_misfit=False, return_params=False):
	"""
	USAGE
	-----
	xm = sinfitn(d, t, periods, constant=True, line=True, return_misfit=False, return_params=False)

	Returns a statistical model 'xm(t)' with M periodic components. The period of each sinusoid
	is specified in the array 'period'. The model parameters are obtained by solving the
	overdetermined least-squares problem that minimizes the model misfit as measured by
	the Euclidean norm L2 = ||Gm - d||^2:

	m = (G^{T}*G)^{-1} * G^{T}*d,

	where 'd(t)'' is the data vector (N x 1) specified at the coordinates 't', m is the model
	parameter vector and G is the data kernel matrix.

	If 'constant'=True (default), a constant term is added to the model. If 'line'=True (default),
	a linear term is added to the model.

	If return_misfit=True (defaults to False), the model misfit L2 is also returned.

	If 'return_params'=True (defaults to False), the model parameter vector 'm' is returned
	instead of the statistical model 'xm'.

	REFERENCES
	----------
	Strang, G.: Introduction to Linear Algebra. 4th edition (2009).

	EXAMPLE
	-------
	>>> import numpy as np
	>>> import matplotlib.pyplot as plt
	>>> from ap_tools.fit import sinfitn
	>>> t = np.linspace(0., 200., 300.)
	>>> periods = [25., 50., 100.]
	>>> f1, f2, f3 = 2*np.pi/np.array(periods)
	>>> x = 50. + 0.8*t + 12.5*sin(f1*t) + 10*sin(f2*t) + 30*sin(f3*t) + 5.0*np.random.randn(t.size)
	>>> xm = sinfitn(x, t, periods)
	>>> fig, ax = plt.subplots()
	>>> ax.plot(t, x, 'g', label='Data')
	>>> ax.plot(t, xm, 'm', label='Sinusoidal model')
	>>> ax.grid(True)
	>>> ax.legend(loc='upper left')
	>>> ax.set_xlabel('Time [s]', fontsize=20)
	>>> ax.set_ylabel('Signal [arbitrary units]', fontsize=20)
	>>> plt.show()
	"""
	d = np.matrix(x)
	periods = list(periods)
	N = d.size       # Number of data points.
	M = len(periods) # Number of model parameters.

	## The data must be a row vector.
	if d.shape==(1,N):
		d = d.T

	## Setting up the data kernel matrix G.
	## Contains model functions evaluated at data points.
	G = np.matrix(np.zeros((N, 0)))
	if constant:
		x0 = np.expand_dims(np.repeat(1., N), 1)
		G = np.concatenate((G,x0), axis=1)
		M = M + 1
	if line:
		x1 = np.expand_dims(t, 1)
		G = np.concatenate((G,x1), axis=1)
		M = M + 1

	for period in periods:
		fn = 2*np.pi/period
		xn = np.matrix([np.sin(fn*t), np.cos(fn*t)]).T
		G = np.concatenate((G,xn), axis=1)

	## Solution to the overdetermined least-squares problem
	## to obtain the model parameters that minimize the
	## L2 norm of the model misfit vector, ||Gm - d||^2.
	## e.g., Strang (2009), pg. 218.
	m = (G.T*G).I*G.T*d

	## Assemble the statistical model using the parameters in the vector m.
	xm = G*m

	## Compute the L2 norm, ||Gm - d||^2.
	err = xm - d
	L2 = np.sqrt(err.T*err)
	L2 = np.float(L2)
	print("")
	print("Model-data misfit: %.1f"%L2)

	if return_params:
		if return_misfit:
			return m, L2
		else:
			return m
	else:
		xm = np.array(xm).squeeze()
		if return_misfit:
			return xm, L2
		else:
			return xm


def fourfilt(x, dts, Tmax, Tmin):
    # See the docstring of the original MATLAB function.
    # The usage is the same, see the docstring for
    # low-pass, high-pass and band-pass filter examples.
    #
    # In the Python version, the variable names
    # have been changed as follows:
    #
    # MATLAB      Python
    #  delt   ->   dts
    #  tmax   ->   Tmax
    #  tmin   ->   Tmin
    # filtdat ->   xfilt
    #
    # Translated from the MATLAB function fourfilt.m
    # by André Palóczy (github.com/apaloczy).
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # function [filtdat]=fourfilt(x,delt,tmax,tmin)
    #  FOURFILT  Fourier low, high, or bandpass filter.
    #
    #     [filtdat]=fourfilt(x,delt,tmax,tmin)
    #
    #     where:   x:      data series to be filtered
    #              delt:   sampling interval
    #              tmax:   maximum period filter cutoff
    #              tmin:   minimum period filter cutoff
    #
    #     usage:  lowpassdata=fourfilt(data,0.5,2000,20)
    #
    #               gives lowpass filter with cutoff at 20.0 sec
    #               tmax set > (length(x)*delt) for no cutoff at low freq end
    #
    #     usage:  highpassdata=fourfilt(x,0.5,20,0.9)
    #
    #               gives highpass filter with cutoff at 20.0 sec
    #               tmin set < (2*delt) for no cutoff at high freq end
    #
    #     usage:  bandpassdata=fourfilt(x,0.5,20,10)
    #
    #               gives bandpass filter passing 10-20 sec. band
    #
    # Reference:
    # Walters, R. A.  and Heston, C., 1982. Removing the tidal-period variations from time-series
    # data using low-pass digital filters. Journal of Physical Oeanography, 12 112-115 .
    #
    #############################
    # Version 1.0  (12/4/96)  Jeff List (jlist@usgs.gov)
    # Version 1.1  (1/8/97)  Rich Signell (rsignell@usgs.gov)
    #     removed argument for number of points and add error trapping for matrices
    # Version 1.1b (12/1/2005) Rich Signell (rsignell@usgs.gov)
    #     added reference
    # (3/18/2019)
    # Translated to Python by André Palóczy.
    #############################
    npts = x.size
    # if npts%2==0: # N is even.
    nby2 = npts//2
    # else:
        # nby2 = (npts-1)//2

    tfund = npts*dts
    ffund = 1.0/tfund

    xmean = x.mean()
    x -= xmean # Remove the mean from data.
    coeffs = fft(x) # Fourier-transform data.

    # Filter coefficients.
    f = ffund
    for i in range(1, nby2+2):
        t = 1.0/f
        if np.logical_or(t>Tmax, t<Tmin):
            coeffs[i] = coeffs[i]*0
        f += ffund

    # Calculate the remaining coefficients.
    for i in range(1, nby2+1):
        coeffs[npts-i] = coeffs[i].conj()

    # Back-transform data and take real part.
    xfilt = ifft(coeffs).real
    xfilt += xmean # Add back the mean.

    return xfilt
