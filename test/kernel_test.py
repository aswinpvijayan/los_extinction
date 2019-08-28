import numpy as np
import sys
from scipy import integrate
from functools import partial

#All the calculations are in units of b/h (impact parameter/smoothing length).

pi = np.pi

def inp_kernel(r, ktype):
    
    """
        `r` is in units of h^-1 where h is the smoothing length of the object in consideration.
        The kernels are in units of h^-3, hence the need to divide by h^2 at the end.
        
        Defined kernels at the moment are `uniform`, `sph-anarchy`, `gadget-2`, `cubic`, `quintic`
    """
    
    if ktype == 'uniform':
        
        if r < 1.:
            return 1./((4./3.)*pi)
        else:
            return 0.
    
    elif ktype == 'sph-anarchy':
        
        if r < 1.: return (21./(2.*pi)) * ((1. - r)*(1. - r)*(1. - r)*(1. - r)*(1. + 4.*r)) 
        else: return 0.       
            
    elif ktype == 'gadget-2':
         
        if r < 0.5: return (8./pi) * (1. - 6*(r*r) + 6*(r*r*r))
        elif r < 1.: return (8./pi) * 2 * ((1. - r)*(1. - r)*(1. - r))
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
    
        print ("Doesn't recognize the kernel. Input your own kernel in `inp_kernel`")
        exit()

     
def integral_func(kern):
    
    return lambda r : 4 * pi * (r*r) * kern(r)


def vol_integral(ktype):
    
    this_kern = partial(inp_kernel, ktype=ktype)

    y, yerr = integrate.quad(integral_func(this_kern), 0, np.inf)
        
    return y
    
    
if __name__== "__main__": 
    """    
    Integrating the kernel over the volume should give 1, i.e.
    integral 4*pi*r^2 W(r)dr from r = 0 to 1
    """
    tol = 1e-5
    y =  vol_integral('cubic') 
    if np.logical_and(y-1.<tol, 1.-y<tol):
    
        print ("\n\t###########\t SUCCESS \t###########")
        
    else:
    
        print ("\n\t###########\t FAIL  \t###########")
