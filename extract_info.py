import sys
sys.path.append('eagle_IO/eagle_IO')
import eagle_IO as E
import numpy as np
import pandas as pd
from HDF5_write import HDF5_write
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from joblib import Parallel, delayed

kinp = np.load('kernel_anarchy.npz')
lkernel = kinp['kernel']
header = kinp['header']
kbins = header.item()['bins']
print('Kernel used is `{}` and the number of bins in the look up table is {}'.format(header.item()['kernel'], kbins))

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

def make_faceon(this_g_cood, this_g_mass, this_g_vel):

    ok = np.where(np.sqrt(np.sum(this_g_cood**2, axis = 1)) <= 0.03)
    this_g_cood = this_g_cood[ok]
    this_g_mass = this_g_mass[ok]
    this_g_vel = this_g_vel[ok]
    
    #Get the angular momentum unit vector
    L_tot = np.array([this_g_mass]).T*np.cross(this_g_cood, this_g_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))
    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag
    
    #z-axis as the face-on axis
    theta = np.arccos(L_unit[2]) 

    vec = np.cross(np.array([0., 0., 1.]), L_unit)
    r = rotation_matrix(vec, theta)

    this_sp_cood *= 1e3
    new = np.array(this_s_cood.dot(r))
    
    return new

def get_SFT(SFT, z):
    
    SFz = (1/SFT) - 1.
    SFz = cosmo.age(z).value - cosmo.age(SFz).value
    return SFz

def get_age(arr, z, n = 4):
    
    Age = Parallel(n_jobs = n)(delayed(get_SFT)(i, z) for i in arr)  

    return np.array(Age)

def get_Z_LOS(cop, s_cood, g_cood, g_mass, g_Z, g_sml):
    
    Z_los_SD = np.zeros(len(s_cood))
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the 
    #particle orientation to face-on 
    xdir, ydir, zdir = 0, 1, 2
    for j in range(len(s_cood)):
        
        
        spos = s_cood[j]
        ok = g_cood[:,zdir] > spos[zdir]
        gpos = g_cood[ok]
        gsml = g_sml[ok]
        gZ = g_Z[ok]
        gmass = g_mass[ok]
        x = gpos[:,xdir] - spos[xdir]
        y = gpos[:,ydir] - spos[ydir]
        
        b = np.sqrt(x*x + y*y)
        
        ok = b <= gsml
        
        boverh = b[ok]/gsml[ok]
        kernel_vals = np.array([lkernel[int(kbins*l)] for l in boverh])

        Z_los_SD[j] = np.sum((gmass[ok]*gZ[ok]/(gsml[ok]*gsml[ok]))*kernel_vals) #in units of Msun/Mpc^2
        
    Z_los_SD *= conv #in units of g/cm^2
    
    return Z_los_SD



def save_to_hdf5(num, tag):
    """
    
    Args:
        num (str): the G-EAGLE id of the sim; eg: '00', '01', ...
        tag (str): the file tag; eg: '000_z015p00', '001_z014p000',...., '011_z004p770'

    """
    
    #num = '00'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_{}/data'.format(num) 
    #tag = '010_z005p000'

    filename = '{}_sp_info.hdf5'.format(tag)


    z = E.read_header('SUBFIND', sim, tag, 'Redshift')
    mstar = E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=4, noH=True, physicalUnits=True)[:,4]*1e10
    indices = np.where(mstar > 10**8)[0]
    sgrpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/SubGroupNumber', numThreads=4)
    grpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/GroupNumber', numThreads=4)
    cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=True, physicalUnits=True, numThreads=4)
    vel = E.read_array('SUBFIND', sim, tag, '/Subhalo/Velocity', noH=True, physicalUnits=True, numThreads=4)

    sp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=4)
    sp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=4)
    sp_mass = E.read_array('PARTDATA', sim, tag, '/PartType4/Mass', noH=True, physicalUnits=True, numThreads=4) * 1e10
    sp_Z = E.read_array('PARTDATA', sim, tag, '/PartType4/Metallicity', numThreads=4)
    sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    sp_vel = E.read_array('PARTDATA', sim, tag, '/PartType4/Velocity', noH=True, physicalUnits=True, numThreads=4)
    sp_sl = E.read_array('PARTDATA', sim, tag, '/PartType4/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)
    sp_ft = E.read_array('PARTDATA', sim, tag, '/PartType4/StellarFormationTime', noH=True, physicalUnits=True, numThreads=4)

    gp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=4)
    gp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=4)
    gp_mass = E.read_array('PARTDATA', sim, tag, '/PartType0/Mass', noH=True, physicalUnits=True, numThreads=4) * 1e10
    gp_Z = E.read_array('PARTDATA', sim, tag, '/PartType0/Metallicity', numThreads=4)
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=4)
    gp_vel = E.read_array('PARTDATA', sim, tag, '/PartType0/Velocity', noH=True, physicalUnits=True, numThreads=4)
    gp_sl = E.read_array('PARTDATA', sim, tag, '/PartType0/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)



    print("Writing out properties for {} subhalos in G-EAGLE_{}".format(len(indices), num))

    for i, j in enumerate(indices):

        s_ok = np.logical_and(sp_sgrpn == sgrpno[j], np.logical_and(sp_grpn == grpno[j], np.sqrt(np.sum((sp_cood-cop[j])**2, axis = 1))<=0.03))
        
        g_ok = np.logical_and(gp_sgrpn == sgrpno[j], np.logical_and(gp_grpn == grpno[j], np.sqrt(np.sum((gp_cood-cop[j])**2, axis = 1))<=0.03))
        
        Z_los_SD = get_Z_LOS(cop[j], sp_cood[s_ok], gp_cood[g_ok], gp_mass[g_ok], gp_Z[g_ok], gp_sl[g_ok])
        
        if i == 0:
        
            hdf5_store = HDF5_write(filename)
            hdf5_store.create_grp(num)

            hdf5_store.create_grp('{}/Subhalo'.format(num))
            hdf5_store.create_grp('{}/Particle'.format(num))
            
            snum = np.array([0, int(np.sum(s_ok))])
            #gnum = np.array([0, int(np.sum(g_ok))])
            
            hdf5_store.create_dset(mstar[j], 'Mstar_30', '{}/Subhalo'.format(num))
            hdf5_store.create_dset(np.array([snum]), 'S_Length', '{}/Subhalo'.format(num))
            #hdf5_store.create_dset(np.array([gnum]), 'G_Length', '{}/Subhalo'.format(num))
            hdf5_store.create_dset(np.array([cop[j]]), 'COP', '{}/Subhalo'.format(num))
            hdf5_store.create_dset(np.array([vel[j]]), 'Velocity', '{}/Subhalo'.format(num))
            
            
            hdf5_store.create_dset(sp_mass[s_ok], 'S_Mass', '{}/Particle'.format(num))
            #hdf5_store.create_dset(gp_mass[g_ok], 'G_Mass', '{}/Particle'.format(num))
            
            hdf5_store.create_dset(sp_Z[s_ok], 'S_Z', '{}/Particle'.format(num))
            #hdf5_store.create_dset(gp_Z[g_ok], 'G_Z', '{}/Particle'.format(num))
            
            #hdf5_store.create_dset(sp_cood[s_ok], 'G_Coordinates', '{}/Particle'.format(num))
            hdf5_store.create_dset(gp_cood[g_ok], 'S_Coordinates', '{}/Particle'.format(num))
            
            #hdf5_store.create_dset(sp_vel[s_ok], 'S_vel', '{}/Particle'.format(num))
            #hdf5_store.create_dset(gp_vel[g_ok], 'G_vel', '{}/Particle'.format(num))
            
            hdf5_store.create_dset(sp_sl[s_ok], 'S_sml', '{}/Particle'.format(num))
            #hdf5_store.create_dset(sp_vel[s_ok], 'G_sml', '{}/Particle'.format(num))
            
            hdf5_store.create_dset(get_age(sp_ft[s_ok], z, 8), 'S_Age', '{}/Particle'.format(num))
            hdf5_store.create_dset(Z_los_SD, 'S_los', '{}/Particle'.format(num))
            
        else:
            snum = np.array([0, int(np.sum(s_ok))]) + snum[1]
            #gnum = np.array([0, int(np.sum(g_ok))]) + gnum[1]
            
            hdf5_store.append(mstar[j], 'Mstar_30', '{}/Subhalo'.format(num))
            hdf5_store.append(np.array([snum]), 'S_Length', '{}/Subhalo'.format(num))
            #hdf5_store.append(np.array([gnum]), 'G_Length', '{}/Subhalo'.format(num))
            hdf5_store.append(np.array([cop[j]]), 'COP', '{}/Subhalo'.format(num))
            hdf5_store.append(np.array([vel[j]]), 'Velocity', '{}/Subhalo'.format(num))
            
            
            hdf5_store.append(sp_mass[s_ok], 'S_Mass', '{}/Particle'.format(num))
            #hdf5_store.append(gp_mass[g_ok], 'G_Mass', '{}/Particle'.format(num))
            
            hdf5_store.append(sp_Z[s_ok], 'S_Z', '{}/Particle'.format(num))
            #hdf5_store.append(gp_Z[g_ok], 'G_Z', '{}/Particle'.format(num))
            
            #hdf5_store.append(sp_cood[s_ok], 'G_Coordinates', '{}/Particle'.format(num))
            hdf5_store.append(gp_cood[g_ok], 'S_Coordinates', '{}/Particle'.format(num))
            
            #hdf5_store.append(sp_vel[s_ok], 'S_vel', '{}/Particle'.format(num))
            #hdf5_store.append(gp_vel[g_ok], 'G_vel', '{}/Particle'.format(num))
            
            hdf5_store.append(sp_sl[s_ok], 'S_sml', '{}/Particle'.format(num))
            #hdf5_store.append(sp_vel[s_ok], 'G_sml', '{}/Particle'.format(num))
            
            hdf5_store.append(get_age(sp_ft[s_ok], z, 8), 'S_Age', '{}/Particle'.format(num))
            hdf5_store.append(Z_los_SD, 'S_los', '{}/Particle'.format(num))




