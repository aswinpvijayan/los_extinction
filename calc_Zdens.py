import eagle as E
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u

conv = (u.solMass/u.kpc**2).to(u.gram/u.cm**2)

def rotation_matrix(axis, theta):
    """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        `https://stackoverflow.com/questions/6802577/rotation-of-3d-vector`
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def make_faceon():

    ok = np.where(np.sqrt(np.sum(this_gp_cood**2, axis = 1)) <= 0.03)
    this_gp_cood = this_gp_cood[ok]
    this_gp_mass = this_gp_mass[ok]
    this_gp_vel = this_gp_vel[ok]
    
    #Get the angular momentum unit vector
    L_tot = np.array([this_gp_mass]).T*np.cross(this_gp_cood, this_gp_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))
    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag
    
    #z-axis as the face-on axis
    theta = np.arccos(L_unit[2]) 

    vec = np.cross(np.array([0., 0., 1.]), L_unit)
    r = rotation_matrix(vec, theta)

    this_sp_cood *= 1e3
    new = np.array(this_sp_cood.dot(r))
    
    return new


def create_3d(coords):

    # Do the plotting in a single call.
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax


kinp = np.load('kernel_anarchy.npz')
lkernel = kinp['kernel']
header = kinp['header']
kbins = header.item()['bins']
print('Kernel used is `{}` and the number of bins in the look up table is {}'.format(header.item()['kernel'], kbins))

num = '00'
sim = '../../G-EAGLE/GEAGLE_{}/data'.format(num) 
tag = '010_z005p000'

#def get_losZsurf(sim, tag):


mstar = E.readArray('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=1)[:,4]*1e10
indices = np.where(mstar > 10**10)[0]
shsgrpno = E.readArray('SUBFIND', sim, tag, '/Subhalo/SubGroupNumber', numThreads=1)
grpno = E.readArray('SUBFIND', sim, tag, '/Subhalo/GroupNumber', numThreads=1)
cop = E.readArray('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', physicalUnits=True, numThreads=1)
vel = E.readArray('SUBFIND', sim, tag, '/Subhalo/Velocity', physicalUnits=True, numThreads=1)

sp_sgrpn = E.readArray('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=1)
sp_grpn = E.readArray('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=1)
sp_mass = E.readArray('PARTDATA', sim, tag, '/PartType4/Mass', numThreads=1) * 1e10
sp_Z = E.readArray('PARTDATA', sim, tag, '/PartType4/Metallicity', numThreads=1)
sp_cood = E.readArray('PARTDATA', sim, tag, '/PartType4/Coordinates', physicalUnits=True, numThreads=1)
sp_vel = E.readArray('PARTDATA', sim, tag, '/PartType4/Velocity', physicalUnits=True, numThreads=1)
sp_sl = E.readArray('PARTDATA', sim, tag, '/PartType4/SmoothingLength', physicalUnits=True, numThreads=1)

gp_sgrpn = E.readArray('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=1)
gp_grpn = E.readArray('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=1)
gp_mass = E.readArray('PARTDATA', sim, tag, '/PartType0/Mass', numThreads=1) * 1e10
gp_Z = E.readArray('PARTDATA', sim, tag, '/PartType0/Metallicity', numThreads=1)
gp_cood = E.readArray('PARTDATA', sim, tag, '/PartType0/Coordinates', physicalUnits=True, numThreads=1)
gp_vel = E.readArray('PARTDATA', sim, tag, '/PartType0/Velocity', physicalUnits=True, numThreads=1)
gp_sl = E.readArray('PARTDATA', sim, tag, '/PartType0/SmoothingLength', physicalUnits=True, numThreads=1)


i = indices[0]

s_ok = np.logical_and(sp_sgrpn == shsgrpno[i], sp_grpn == grpno[i])
g_ok = np.logical_and(gp_sgrpn == shsgrpno[i], gp_grpn == grpno[i])


this_gp_cood = gp_cood[g_ok]
this_gp_mass = gp_mass[g_ok]
this_gp_Z = gp_Z[g_ok]
this_gp_vel = gp_vel[g_ok]
this_gp_sl = gp_sl[g_ok]
this_gp_cood -= cop[i]
this_gp_vel -= vel[i]
this_sp_cood = sp_cood[s_ok] - cop[i]
this_sp_sl = sp_sl[s_ok]
Z_los_SD = np.zeros(len(this_sp_cood))
#Fixing the observer direction as z-axis. Use make_faceon() for changing the 
#particle orientation to face-on 
xdir, ydir, zdir = 0, 1, 2
for j in range(len(this_sp_cood)):

    spos = this_sp_cood[j]
    ssl = this_sp_sl[j]
    ok = (this_gp_cood[:,zdir] > spos[zdir])
    gpos = this_gp_cood[ok]
    gsl = this_gp_sl[ok]
    gZ = this_gp_Z[ok]
    gmass = this_gp_mass[ok]
    x = gpos[:,xdir] - spos[xdir]
    y = gpos[:,ydir] - spos[ydir]
    
    b = np.sqrt(x*x + y*y)
    
    ok = (b <= gsl)
    
    boverh = b[ok]/gsl[ok]
    kernel_vals = np.array([lkernel[int(kbins*l)] for l in boverh])

    Z_los_SD[j] = np.sum(gmass[ok]*gZ[ok]/(gsl[ok]*gsl[ok])*kernel_vals) #in units of Msun/Mpc^2
    
Z_los_SD *= conv #in units of g/cm^2
"""
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), sharex=True, sharey=True, facecolor='w', edgecolor='k')
x, y = this_sp_cood[:,0]*1e3, this_sp_cood[:,1]*1e3
dx = min(this_sp_sl)*1e3
gridsize = np.array([(int(max(x)-min(x))/dx), int((max(y)-min(y))/dx)]).astype(int)
p = axs.hexbin(x, y, gridsize=gridsize, bins = 'log', alpha = 0.7, C = Z_los_SD, cmap = plt.cm.get_cmap('jet'), mincnt = 1)
axs.set_xlim(-5, 5)
axs.set_ylim(-5, 5)
plt.show()
"""
