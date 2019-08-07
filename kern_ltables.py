import numpy as np
import sys
from scipy import integrate
from functools import partial

#All the calculations are in units of b/h (impact parameter/smoothing length).

"""
    Follwing the same method used in `pyloser` (https://bitbucket.org/romeeld/pyloser/src/default/) 

    Line of sight distance along a particle, l = sqrt(h^2 + b^2), where h and b 
    are the smoothing length and the impact parameter respectively. This needs 
    to be weighted along with the kernel density function W(r), to calculate the 
    los density. Integrated los density, D = 2 * integral(W(r)dl) from 0 to h, 
    where r = sqrt(x^2 + b^2), W(r) is in units of f(r) h^-3 where f(r) is some 
    function of r. 

    The parameters are normalized in terms of the smoothing length, helping us to 
    create a look-up table for every impact parameter along the line-of-sight.
    Hence we substitute x = x/h and b = b/h.  
    dl = xdx/sqrt(x^2 + b^2) * h
    
    This implies
    D = h^-2 * 2 * integral(f(r) dl) for x = 0 to 1.

    The division by h^2 is to be done separately for each particle along the line-of-sight.
"""

kernsize = 10000
pi = np.pi

def inp_kernel(r, ktype):
    
    """
        `r` is in units of h^-1 where h is the smoothing length of the object in consideration.
        The kernels are in units of h^-3, hence the need to divide by h^2 at the end.
        
        Defined kernels at the moment are `uniform`, `anarchy`, `gadget-2`, `cubic`, `quintic`
    """
    
    if ktype == 'uniform':
        
        if r < 1.:
            return 1./((4./3.)*np.pi)
        else:
            return 0.
    
    elif ktype == 'anarchy':
        
        if r < 1.: return (21./(2.*pi)) * ((1. - r)*(1. - r)*(1. - r)*(1. - r)*(1. + 4.*r)) 
        else: return 0.       
            
    elif ktype == 'gadget-2':
        
        if r < 0.5: return (8./pi) * (1. - 6*(r*r) + 6*(r*r*r))
        elif r < 1.: return (8./pi) * ((1. - r)*(1. - r)*(1. - r))
        else: return 0.
    
    elif ktype == 'cubic':
        
        if r < 0.5: return (2.546479089470 + 15.278874536822 * (r - 1.0) * r * r)
        elif r < 1: return 5.092958178941 * (1.0 - r) * (1.0 - r) * (1.0 - r)
        else: return 0
        
    elif ktype == 'quintic':
        
        if r < 0.333333333: return 27.0*(6.4457752*r*r*r*r*(1.0-r) -1.4323945*r*r +0.17507044)
        elif r < 0.666666667: return 27.0*(3.2228876*r*r*r*r*(r-3.0) +10.7429587*r*r*r -5.01338071*r*r +0.5968310366*r +0.1352817016)
        elif r < 1: return 27.0*0.64457752*(-r*r*r*r*r +5.0*r*r*r*r -10.0*r*r*r +10.0*r*r -5.0*r +1.0)
        else: return 0
            
    else:
    
        print ("Doesn't recognize the kernel. Input your own kernel in `inp_kernel` in `kern_ltables.py`")
        exit()

     
def fr_dl(x, kern, b):
    
    """
        f(r)dl
    """
    
    ds = np.sqrt(b*b + x*x) 
    
    return (x/ds) * kern(ds) 

def integral_func(kern, i):

    return lambda x : fr_dl(x, kern, i)

def get_kernel(ktype):
    
    """
        h^-2 * 2 * integral(f(r) dl) from x = 0 to 1 for various values of `b`
    """
    
    kernel = np.zeros(kernsize + 1)
    this_kern = partial(inp_kernel, ktype=ktype)

    bins = np.arange(0, 1., 1./kernsize)
    bins = np.append(bins, 1.)

    for i in range(kernsize):

        y, yerr = integrate.quad(integral_func(this_kern, bins[i]), 0, 1.) 
        kernel[i] = y * 2.
        
    return kernel
    
def create_kernel(ktype='anarchy'):
    
    """
        Saves the computed kernel for easy look-up as .npz file
    """
    
    kernel = get_kernel(ktype)
    header = np.array(['kernel': ktype, 'bins': kernsize])
    np.savez('kernel_{}.npz'.format(ktype), header=header, kernel=kernel)
    
    print (header)
    
    return kernel
    
    


