# -*- coding: utf-8 -*-
#
# Description: Loose functions to perform statistical calculations.
#              These functions were written partly for study purposes
#              and are probably available in other modules in a more
#              efficient form.
#
# Author:      André Palóczy
# E-mail:      paloczy@gmail.com

from __future__ import division

__all__ = ['gauss_curve',
		   'principal_ang',
		   'nmoment',
		   'skewness',
		   'kurtosis',
		   'rcoeff',
		   'autocorr',
		   'Tdecorr',
		   'Neff',
		   'lnsmc',
		   'ci_mean',
		   'rsig',
		   'rsig_student',
		   'rci_fisher',
		   'rci_boot']

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv, betainc, betaincinv
from scipy.stats.distributions import norm
from scipy.stats.distributions import t as student
from ap_tools.utils import near
try:
	from scikits import bootstrap
except:
	print("scikits.bootstrap not installed. Function 'rci_boot' will not be available.")

def gauss_curve(r, xmean, xstd):
	"""
	USAGE
	-----
	g = gauss_curve(r, xmean, xstd)

	Returns an exact Gaussian curve that corresponds to the
	Probability Density Function (PDF) of a normally-distributed
	random variable x(r) with mean 'xmean' and standard deviation 'xstd'.

	References
	----------
	TODO
	"""
	return np.exp(-(r-xmean)**2/(2*xstd**2))/(xstd*np.sqrt(2*np.pi))

def principal_ang(x, y):
	"""
	USAGE
	-----
	ang = principal_ang(x, y)

	Calculates the angle that the principal axes of two random variables
	'x' and 'y' make with the original x-axis. For example, if 'x' and 'y'
	are two orthogonal components of a vector quantity, the returned angle
	is the direction of maximum variance.

	References
	----------
	TODO
	"""
	x, y = map(np.asanyarray, (x,y))
	assert x.size==y.size

	N = x.size
	x = x - x.mean()
	y = y - y.mean()
	covxy = np.sum(x*y)/N
	varx = np.sum(x**2)/N
	vary = np.sum(y**2)/N
	th_princ = 0.5*np.arctan(2*covxy/(varx-vary))

	return th_princ

def nmoment(x, n=1):
	"""
	USAGE
	-----
	mn = nmoment(x, n=1)

	Calculates the 'n'-th moment (default=1)
	of a random variable 'x'.

	References
	----------
	TODO
	"""
	x = np.asanyarray(x)

	N = x.size
	m1 = np.sum(x)/N
	if n==1: # First moment (mean), do not demean.
		m = m1
	else: # Higher-order moments, demean.
		xp = x - m1
		m = np.sum(xp**n)/N

	return m

def skewness(x):
	"""
	USAGE
	-----
	S = skewness(x)

	Calculates the skewness of a random variable 'x'.
	Skewness is a dimensionless quantity proportional to the
	third moment (m3) of a random variable. It is defined as

	Skew(x) = m3/std**3,

	where 'std' is the square root of the second moment of 'x',
	i.e., the standard deviation of 'x'.

	References
	----------
	TODO
	"""
	x = np.asanyarray(x)

	m3 = nmoment(x, 3)
	m2 = nmoment(x, 2)
	skew = m3/np.sqrt(m2)**3

	return skew

def kurtosis(x):
	"""
	USAGE
	-----
	K = kurtosis(x)

	Calculates the kurtosis of a random variable 'x'.
	Kurtosis is a dimensionless quantity proportional to the
	fourth moment (m4) of a random variable. It is defined as

	Kurt(x) = m4/m2**2,

	where 'm2' is the square of the second moment of 'x',
	i.e., the square of the variance of 'x'.

	References
	----------
	TODO
	"""
	x = np.asanyarray(x)

	m4 = nmoment(x, 4)
	m2 = nmoment(x, 2)
	kurt = m4/m2**2

	return kurt

def rcoeff(x, y):
	"""
	USAGE
	-----
	r = rcoeff(x, y)

	Computes the Pearson correlation coefficient r between series x and y.

	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 257, equation 3.97a.
	"""
	x,y = map(np.asanyarray, (x,y))

	# Sample size.
	assert x.size==y.size
	N = x.size

	# Demeaned series.
	x = x - x.mean()
	y = y - y.mean()

	# Standard deviations.
	sx = x.std()
	sy = y.std()

	## Covariance between series. Choosing unbiased normalization (N-1).
	Cxy = np.sum(x*y)/(N-1)

	## Pearson correlation coefficient r.
	r = Cxy/(sx*sy)

	return r

def autocorr(x, biased=True):
	"""
	USAGE
	-----
	Rxx = autocorr(x, biased=True)

	Computes the biased autocorrelation function Rxx for sequence x,
	if biased==True (default). "biased" means that the k-th value of Rxx is
	always normalized by the total number of data points N, instead of the number
	of data points actually available to be summed at lag k, i.e., (N-k). The
	biased autocorrelation function will therefore always converge to 0
	as the lag approaches N*dt, where dt is the sampling interval.

	If biased==False, compute the unbiased autocorrelation function
	(i.e., normalize by (N-k)).

	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 429, equation 5.4.

	Gille lecture notes on data analysis, available
	at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf
	"""
	x = np.asanyarray(x)

	N = x.size # Sample size.
	Cxx = np.zeros(N)

	# Calculate the mean of the sequence to write in the more intuitive
	# summation notation in the for loop below (less efficient).
	xb = x.mean()

	## Summing for lags 0 through N (the size of the sequence).
	for k in range(N):
		Cxx_aux = 0.
		for i in range(N-k):
			Cxx_aux = Cxx_aux + (x[i] - xb)*(x[i+k] - xb)

		# If biased==True, Calculate BIASED autocovariance function,
		# i.e., the value of Cxx at k-th lag is normalized by the
		# TOTAL amount of data points used (N) at all lags. This weights
		# down the contribution of the less reliable points at greater lags.
		#
		# Otherwise, the value of Cxx at the k-th lag is normalized
		# by (N-k), i.e., an UNBIASED autocovariance function.
		if biased:
			norm_fac = N
		else:
			norm_fac = N - abs(k)
		Cxx[k] = Cxx_aux/norm_fac

	# Normalize the (biased or unbiased) autocovariance
	# function Cxx by the variance of the sequence to compute
	# the (biased or unbiased) autocorrelation function Rxx.
	Rxx = Cxx/np.var(x)

	return Rxx

def Tdecorr(Rxx, M=None, dtau=1.):
	"""
	USAGE
	-----
	Td = Tdecorr(Rxx)

	Computes the integral scale Td (AKA decorrelation scale, independence scale)
	for a data sequence with autocorrelation function Rxx. 'M' is the number of
	lags to incorporate in the summation (defaults to all lags) and 'dtau' is the
	lag time step (defaults to 1).

	The formal definition of the integral scale is the total area under the
	autocorrelation curve Rxx(tau):

                 /+inf
	Td = 2 * |     Rxx(tau) dtau
	         /0

	In practice, however, Td may become unrealistic if all of Rxx is summed
	(e.g., often goes to zero for data dominated by periodic signals); a
	different approach is to instead change M in the summation and use the
	maximum value of the integral Td(t):

                    /t
	Td(t) = 2 * |     Rxx(tau) dtau
	            /0

	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 274, equation 3.137a.

	Gille lecture notes on data analysis, available
	at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf
	"""
	Rxx = np.asanyarray(Rxx)
	C0 = Rxx[0]
	N = Rxx.size # Sequence size.

	# Number of lags 'M' to incorporate in the summation.
	# Sum over all of the sequence if M is not chosen.
	if not M:
		M = N

	# Integrate the autocorrelation function.
	Td = np.zeros(M)
	for m in range(M):
		Tdaux = 0.
		for k in range(m-1):
			Rm = (Rxx[k] + Rxx[k+1])/2. # Midpoint value of the autocorrelation function.
			Tdaux = Tdaux + Rm*dtau # Riemann-summing Rxx.

		Td[m] = Tdaux

	# Normalize the integral function by the autocorrelation at zero lag
	# and double it to include the contribution of the side with
	# negative lags (C is symmetric about zero).
	Td = (2./C0)*Td

	print("")
	print("Theoretical integral scale --> 2 * int 0...+inf [Rxx(tau)] dtau: %.2f."%Td[-1])
	print("")
	print("Maximum value of the cumulative sum (empirical integral scale): %.2f."%Td.max())

	return Td

def Neff(Tdecorr, N, dt=1.):
	"""
	USAGE
	-----
	neff = Neff(Tdecorr, N, dt=1.)

	Computes the number of effective degrees of freedom 'neff' in a
	sequence with integral scale 'Tdecorr' and 'N' data points
	separated by a sampling interval 'dt'.

	Neff = (N*dt)/Tdecorr = (Sequence length)/(Integral scale)

	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 274, equation 3.138.
	"""
	neff = (N*dt)/Tdecorr # Effective degrees of freedom.

	print("")
	print("Neff = %.2f"%neff)

	return neff

def lnsmc(xvar, func, *args, nmc=10000, alpha=0.95, nbins=100, plot_pdf=False, verbose=True):
	"""
	USAGE
	-----
	yvar_err, Yvar_mc = lnsmc(xvar, func, *args, nmc=10000, alpha=0.95, nbins=100, plot_pdf=True, verbose=True)

	Runs 'nmc' (defaults to 10000) Monte Carlo simulations and calculates the
	standard error propagated from the given variable 'xvar' (assumed to be
	normally-distributed) to a derived variable 'yvar', such that

	yvar = func(xvar),

	where 'func' is the specified function relating the two variables.
	'args' is a tuple containing the additional input arguments required
	by 'func', if any.

	The propagated standard error is calculated by numerically integrating
	the PDF of the simulated 'yvar' to estimate its CDF, and finding the
	percentile associated with an 'alpha' confidence level (defaults to 0.95).

	If 'xvar' is a single number, it is assumed to be the prescribed
	standard deviation of 'xvar', i.e., like a nominal instrumental error
	of a sensor.

	'func' is a callable that takes the source variable and outputs the
	derived variable, which is the variable we want to propagate the error to.
	'nbins' is the number of bins to be used for integrating the PDF.
	'plot_pdf' Determines whether or not to plot the PDF of the simulated
	derived variable (defaults to False).
	"""
	assert callable(func), "'func' is not a function."
	xvar = np.array(xvar)

	Yvar_mc = []
	# If 'xvar' is a single number, take it to be
	# a prescribed standard deviation for 'xvar', e.g.,
	# like the instrumental error of a sensor.
	if xvar.size>1:
		xvar = np.array(xvar)
		N = xvar.size
		xvar_mean = xvar.mean()
		xvar_std = xvar.std()
	else:
		N = 1
		xvar_mean = 0.
		xvar_std = xvar

	for n in range(nmc):
		if verbose:
			print("Monte Carlo simulation %d of %d"%(n+1,nmc))
		# Generate two random normal time series.
		xvar_mc = xvar_std*np.random.randn(N) + xvar_mean
		# Calculate derived variable with the random data.
		Yvar_mc.append(func(xvar_mc, *args))

	# Calculate PDF and CDF of the simulated data.
	fig, ax = plt.subplots()
	yvar_n, bins, _ = ax.hist(np.array(Yvar_mc).ravel(), bins=nbins, normed=True, color='b', histtype='bar')

	# Calculate and plot CDF.
	dbin = bins[1:] - bins[:-1]
	cdf = np.cumsum(yvar_n*dbin) # CDF [unitless].
	binm = 0.5*(bins[1:] + bins[:-1])
	binm = np.insert(binm, 0, 0)
	cdf = np.insert(cdf, 0, 0)
	# cdfnear_alpha = near(cdf, alpha)
	# fci_alpha = np.where(cdf==cdfnear_alpha)[0][0]
	fci_alpha = near(cdf, alpha, return_index=True)

	# Value of the Monte Carlo derived variable associated with the alpha*100 percentile.
	yvar_err = binm[fci_alpha]

	if plot_pdf:
		ax2 = ax.twinx()
		ax2.plot(binm, cdf, 'k', linewidth=3.0)
		ax2.axhline(y=alpha, linewidth=1.5, linestyle='dashed', color='grey')
		ax2.axvline(x=yvar_err, linewidth=1.5, linestyle='dashed', color='grey')
		ax.grid()
		ax.set_ylabel(r'Probability density', fontsize=18, fontweight='black')
		ax2.set_ylabel(r'Cumulative probability', fontsize=18, fontweight='black')
		ax.set_xlabel(r'Derived variable', fontsize=18, fontweight='black')
		ytks = ax2.get_yticks()
		ytks = np.append(ytks, alpha)
		ytks.sort()
		ytks = ax2.set_yticks(ytks)
		fig.canvas.draw()
		plt.show()
	else:
		plt.close()

	# Error is symmetric about zero, because the
	# derived variable is assumed to be normal.
	yvar_err = yvar_err/2.
	if verbose:
		print("The Monte Carlo error for the derived variable is +-%.2f (alpha=%.2f)"%(yvar_err, alpha))
	# Also return the values of the derived variable for
	# all Monte Carlo simulations.
	#
	# yvar_err is the envelope containing 100*'alpha' % of the values
	# of the simulated derived variable. So the error bars
	# are +-yvar_err/2 (symmetric about each data point).
	return yvar_err, Yvar_mc

def ci_mean(m_sample, s_sample, ndof_eff, alpha=0.95, verbose=True):
	"""
	Calculates a confidence interval at the 'alpha' significance level
	for the sample mean of a normally-distributed random variable with
	sample mean 'm_sample', sample standard deviation 's_sample' and
	effective degrees of freedom 'ndof_eff'.

	References
	----------
	TODO

	Example
	-------
	TODO
	"""
	# z-score of the CI associated with the given significance level 'alpha'.
	# Standard normal curve is symmetric about the mean (0), therefore take
	# only the upper CI.
	zs = norm.interval(alpha)[1]
	# Lower (upper) 100 * alpha % confidence interval.
	std_err = s_sample/np.sqrt(ndof_eff)
	xl = m_sample - zs*std_err
	xu = m_sample + zs*std_err

	if verbose:
		print("")
		print("Sample mean CI (xl,xu): (%.3f,%.3f)"%(xl, xu))
		print("")

	return (xl,xu)

def rsig(ndof_eff, alpha=0.95):
	"""
	USAGE
	-----
	Rsig = rsig(ndof_eff, alpha=0.95)

	Computes the minimum (absolute) threshold value 'rsig' that
	the Pearson correlation coefficient r between two normally-distributed
	data sequences with 'ndof_eff' effective degrees of freedom has to have
	to be statistically significant at the 'alpha' (defaults to 0.95)
	confidence level.

	For example, if rsig(ndof_eff, alpha=0.95) = 0.2 for a given pair of
	NORMALLY-DISTRIBUTED samples with a correlation coefficient r>0.7, there
	is a 95 % chance that the r estimated from the samples is significantly
	different from zero. In other words, there is a 5 % chance that two random
	sequences would have a correlation coefficient higher than 0.7.

	OBS: This assumes that the two data series have a normal distribution.

	Translated to Python from the original matlab code by Prof. Sarah Gille
	(significance.m), available at http://www-pord.ucsd.edu/~sgille/sio221c/

	References
	----------
	Gille lecture notes on data analysis, available
	at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf

	Example
	-------
	TODO
	"""
	rcrit_z = erfinv(alpha)*np.sqrt(2./ndof_eff)

	return rcrit_z

def rsig_student(ndof_eff, alpha=0.95):
	"""
	USAGE
	-----
	Rsigt = rsig_student(ndof_eff, alpha=0.95)

	References
	----------
	https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

	Example
	-------
	TODO
	"""
	ndof = ndof_eff - 2

	## Find the critical value of r from the Student t distribution
	## by inverting the survival function (1-CDF).
	pval = 1 - alpha
	tcrit = student.isf(pval,ndof)

	## Convert the critical value of the t statistic
	## into a critical value of r.
	rcrit_t = tcrit/np.sqrt(ndof + tcrit**2)

	return rcrit_t

def rci_fisher(r, ndof_eff, alpha=0.95, verbose=True):
	"""
	Calculate a confidence interval for the Pearson correlation coefficient r
	between two series 'x' and 'y' using the Fisher's transformation method.

	References
	----------
	Cox (2008): Speaking Stata: Correlation with confidence, or Fisher’s z revisited.
	The Stata Journal (2008) 8, Number 3, pp. 413-439.

	https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

	Example
	-------
	TODO
	"""
	## OBS: equivalent to the form z_r = 0.5*np.log((1+r)/(1-r))
	## which is also commonly found in the literature.
	z_r = np.arctanh(r)   # Transform r to a standard normal variable z_r.
	se_r = 1./np.sqrt(ndof_eff-3) # Standard error of the transformed distribution.

	# z-score of the CI associated with the given significance level 'alpha'.
	# Standard normal curve is symmetric about the mean (0), therefore take
	# only the upper CI.
	zs = norm.interval(alpha)[1]
	# Lower (upper) 100 * alpha % confidence interval.
	z_xl = z_r - zs*se_r
	z_xu = z_r + zs*se_r

	## Use the inverse transformation to convert intervals
	## in the z-scale back to the r-scale.
	xl = np.tanh(z_xl)
	xu = np.tanh(z_xu)

	if verbose:
		print("")
		print("Fisher transform CI (xl,xu): (%.3f,%.3f)"%(xl, xu))
		print("")

	return (xl,xu)

def rci_boot(x, y, alpha=0.95, verbose=True, n_samples=10000, method='bca'):
	"""
	Calculate a confidence interval for the Pearson correlation coefficient r
	between two series 'x' and 'y' using the bootstrap method. It is helpful to
	compare the bootstrapped Confidence Intervals (CIs) for the Pearson correlation
	coefficient r with the CIs obtained with the more standard Fisher’s transformation
	method, as suggested by Cox (2008).

	References
	----------
	Cox (2008): Speaking Stata: Correlation with confidence, or Fisher’s z revisited.
	The Stata Journal (2008) 8, Number 3, pp. 413-439.

	Example
	-------
	TODO
	"""
	x,y = map(np.asanyarray, (x,y))
	## Bootstrapped confidence intervals.
	xl, xu = bootstrap.ci((x, y), statfunction=rcoeff, alpha=(1-alpha), n_samples=n_samples, method=method, multi=True)

	if verbose:
		print("")
		print("Bootstrapped CI (xl,xu): (%.3f,%.3f)"%(xl, xu))
		print("")

	return (xl,xu)
