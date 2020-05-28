import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
import h5py
import schwimmbad
from functools import partial
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
import sys
sys.path.append('./eagle_IO/eagle_IO')
import eagle_IO as E
from numba import jit, njit


conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

@jit
def sphere(coords, a, b, c, r):

    #Equation of a sphere

    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    return (x-a)**2 + (y-b)**2 + (z-c)**2 - r**2

class flares:

    def __init__(self):

        self.halos = np.arange(40)

        self.tags = np.array(['000_z015p000','001_z014p000','002_z013p000',
                              '003_z012p000','004_z011p000','005_z010p000',
                              '006_z009p000','007_z008p000','008_z007p000',
                              '009_z006p000','010_z005p000','011_z004p770'])

        self.radius = 14. #in cMpc/h

        #Put down the sim root location here
        self.directory = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_'
        self.ref_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
        self.agn_directory = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/S15_AGNdT9/data/'

        self.ref_tags = np.array(['001_z015p132','002_z009p993','003_z008p988',
                                  '004_z008p075','005_z007p050','006_z005p971',
                                  '007_z005p487','008_z005p037','009_z004p485'])

        self.data = "%s/data"%(os.path.dirname(os.path.realpath(__file__)))

        ## update with weights file name
        self.weights = './weights.txt'


    def check_snap_exists(self,halo,snap):

        test_str = self.directory + halo + '/data/snapshot_' + snap
        return os.path.isdir(test_str)


    def spherical_region(self, sim, snap):

        """
        Inspired from David Turner's suggestion
        """

        dm_cood = E.read_array('PARTDATA', sim, snap, '/PartType1/Coordinates', noH=False, physicalUnits=False, numThreads=4)  #dm particle coordinates

        hull = ConvexHull(dm_cood)

        cen = [np.median(dm_cood[:,0]), np.median(dm_cood[:,1]), np.median(dm_cood[:,2])]
        pedge = dm_cood[hull.vertices]  #edge particles
        y_obs = np.zeros(len(pedge))
        p0 = np.append(cen, self.radius)

        popt, pcov = curve_fit(sphere, pedge, y_obs, p0, method = 'lm', sigma = np.ones(len(pedge))*0.001)
        dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis = 1))
        centre, radius, mindist = popt[:3], popt[3], np.min(dist)

        return centre, radius, mindist

    def calc_df(self, mstar, tag, volume, massBinLimits):

        hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
        hist = np.float64(hist)
        phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
        phi_sigma = (np.sqrt(hist) / volume) /\
                    (massBinLimits[1] - massBinLimits[0]) # Poisson errors

        return phi, phi_sigma, hist


    def plot_df(self, ax, phi, phi_sigma, hist, massBins,
                label, color, hist_lim=10, lw=3, alpha=0.7):

        mask = (hist >= hist_lim)
        ax.errorbar(np.log10(massBins[mask][phi[mask] > 0.]),
                np.log10(phi[mask][phi[mask] > 0.]),
                yerr=[np.log10(phi[mask][phi[mask] > 0.]+phi_sigma[mask][phi[mask] > 0.]) \
                        - np.log10(massBins[mask][phi[mask] > 0.]),
                      np.log10(massBins[mask][phi[mask] > 0.]) \
                        - np.log10(phi[mask][phi[mask] > 0.]-phi_sigma[mask][phi[mask] > 0.])],
                label=label, lw=lw, c=color, alpha=alpha)

        i = np.where(hist >= hist_lim)[0][-1]
        ax.errorbar(np.log10(massBins[i:][phi[i:] > 0.]),
                np.log10(phi[i:][phi[i:] > 0.]),
                yerr=[np.log10(phi[i:][phi[i:] > 0.] + phi_sigma[i:][phi[i:] > 0.]) - \
                        np.log10(massBins[i:][phi[i:] > 0.]),
                      np.log10(massBins[i:][phi[i:] > 0.]) - \
                              np.log10(phi[i:][phi[i:] > 0.] - phi_sigma[i:][phi[i:] > 0.])],
                lw=lw, linestyle='dotted', c=color, alpha=alpha)



    """
    Utilities for loading and saving nested dictionaries recursively

    see https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    """

    def save_dict_to_hdf5(self, dic, filename, groupname='default', overwrite=False):
        """
        ....
        """
        if groupname=='default': print("Saving to `default` group")
        with h5py.File(filename, 'a') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, groupname+'/', dic, overwrite=overwrite)

    def recursively_save_dict_contents_to_group(self, h5file, path, dic, overwrite=False):
        """
        ....
        """
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, float)):
                if overwrite:
                    old_data = h5file[path + key]
                    old_data[...] = item
                else:
                    h5file[path + key] = item
            elif isinstance(item, dict):
                self.recursively_save_dict_contents_to_group(h5file, path + key + '/', item, overwrite=overwrite)
            else:
                raise ValueError('Cannot save %s type'%type(item))

    def load_dict_from_hdf5(self, filename, group=''):
        """
        ....
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, group+'/')


    def recursively_load_dict_contents_from_group(self, h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()] # .value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file, path + key + '/')
        return ans


def get_SFT(SFT, redshift):

    SFz = (1/SFT) - 1.
    SFz = cosmo.age(redshift).value - cosmo.age(SFz).value
    return SFz

def get_age(arr, z, numThreads = 4):

    if numThreads == 1:
        pool = schwimmbad.SerialPool()
    elif numThreads == -1:
        pool = schwimmbad.MultiPool()
    else:
        pool = schwimmbad.MultiPool(processes=numThreads)

    calc = partial(get_SFT, redshift = z)
    Age = np.array(list(pool.map(calc,arr)))

    return Age


@njit()
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in g/cm^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n = len(s_cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 0, 1, 2
    for ii in range(n):

        thisspos = s_cood[ii]
        ok = (g_cood[:,zdir] > thisspos[zdir])
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = (boverh <= 1.)

        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2

    Z_los_SD*=conv #in units of Msun/pc^2

    return Z_los_SD
