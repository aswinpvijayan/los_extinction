import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import astropy.units as u

conv = (u.Mpc**-2).to(u.pc**-2)

kinp = np.load('../kernel_{}.npz'.format('sph-anarchy'), allow_pickle=True)
lkernel = kinp['kernel']
header = kinp['header']
kbins = header.item()['bins']
print('\n\t\t###################################\nKernel used is `{}` and the number of bins in the look up table is {}\n\t\t###################################\n'.format(header.item()['kernel'], kbins))

@njit(parallel=True)
def get_Zlos(s,sx,sy,gsml,gmass,gZ):
    
    Z_los_SD = np.zeros(s)
    for i in range(s):

        x = gx - sx[i]
        y = gy - sy[i]
        
        b = np.sqrt(x**2 + y**2)
        boverh = b/gsml
        
        ok = np.where(boverh<=1.)[0]
        
        kernvals = np.array([lkernel[int(l)] for l in boverh[ok]])
        
        Z_los_SD[i] = np.sum((gmass[ok]*gZ[ok]/(gsml[ok]*gsml[ok]))*kernvals)
        
    Z_los_SD *= conv
    
    return Z_los_SD


#Fake Stars
s = 10000
r = np.sqrt(10**np.random.uniform(-7,np.log10(35**2),s))
theta = np.pi * np.random.uniform(0,2,s)

sx = r*np.cos(theta)
sy = r*np.sin(theta)
sdist = r

#Gas particles
g = 1000000
r = np.sqrt(10**np.random.uniform(-7,np.log10(40**2),g))
theta = np.pi * np.random.uniform(0,2,g)

gx = r*np.cos(theta)
gy = r*np.sin(theta)
gsml = 15.*np.random.uniform(0,1,g)/1e3
gmass = 1.8e6*np.ones(g)
gZ = 0.0004*np.random.uniform(0,1,g)
gdist = r


Z_los_SD = get_Zlos(s,sx,sy,gsml,gmass,gZ)

aperture = np.logspace(-3.5, np.log10(40.05), 1000)
g_SD = np.zeros(len(aperture))

for ii in range(len(aperture)):

    if ii != 0:
        ok = np.logical_and(gdist<=aperture[ii], gdist>=aperture[ii-1])
        area = np.pi*((aperture[ii]*1e6)**2 - (aperture[ii-1]*1e6)**2)
    else:
        ok = np.where(gdist<aperture[ii])
        area = np.pi*(aperture[ii]*1e6)**2
        
    g_SD[ii] = np.sum(gmass[ok]*gZ[ok])/area
     

fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 12), sharex=False, sharey=False, facecolor='w', edgecolor='k')
axs = axs.ravel()

axs[0].scatter(sx, sy, s = 0.4, label = 'P-stars')
axs[1].scatter(gx, gy, s = 0.4, label = 'P-gas')

gridsize = np.array([int((np.max(sx)-np.min(sx))/0.5), int((np.max(sy)-np.min(sy))/0.5)])
ok = np.where(Z_los_SD>0)
axs[2].hexbin(sx[ok], sy[ok], C = np.log10(Z_los_SD[ok]), gridsize = gridsize, cmap = plt.cm.get_cmap('jet'), mincnt = 1)

for jj in range(3):
    axs[jj].set_xlabel('x/kpc', fontsize = 14)
    axs[jj].set_ylabel('y/kpc', fontsize = 14)
 
axs[3].scatter(np.log10(sdist), np.log10(Z_los_SD), s = 0.5, label = 'S-los')
axs[3].scatter(np.log10(aperture[np.where(g_SD>0)]), np.log10(g_SD[np.where(g_SD>0)]), s = 0.8, label = 'G-SD')
axs[3].set_xlabel(r'log$_{10}$(r/kpc)', fontsize = 14)
axs[3].set_ylabel(r'log$_{10}$($\Sigma_{Z-los}$, $\Sigma_{Z-gas}$/(M$_{\odot}$pc$^{-2}$))', fontsize = 14)


for jj in range(4):
    axs[jj].legend(fontsize=14, loc=1, frameon=False)
    axs[jj].grid()
plt.savefig('convergence_test.png')
plt.show()


