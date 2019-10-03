import numpy as np
from numba import jit, njit
from functools import partial
from astropy import units as u
import schwimmbad
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import sys
sys.path.append('./eagle_IO/eagle_IO')
import gc
import eagle_IO as E
from HDF5_write import HDF5_write
from astropy.cosmology import Planck13 as cosmo
import timeit
from mpi4py import MPI

norm = np.linalg.norm
conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

@jit()
def sphere(coords, a, b, c, r):
    
    #Equation of a sphere

    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    
    return (x-a)**2 + (y-b)**2 + (z-c)**2 - r**2


def fit_sphere(sim, tag, r):
        
        dm_cood = E.read_array('PARTDATA', sim, tag, '/PartType1/Coordinates', noH=False, physicalUnits=False, numThreads=4)
        
        hull = ConvexHull(dm_cood)

        cen = [np.median(dm_cood[:,0]), np.median(dm_cood[:,1]), np.median(dm_cood[:,2])]

        pedge = dm_cood[hull.vertices] 

        y_obs = np.zeros(len(pedge))
        p0 = np.append(cen, r)

        popt, pcov = curve_fit(sphere, pedge, y_obs, p0, method = 'lm', sigma = np.ones(len(pedge))*0.001)
        
        dist = np.sqrt(np.sum((pedge-popt[:3])**2, axis = 1))
        
        return popt[:3], popt[3], np.min(dist) 


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

def make_faceon(cop, this_g_cood, this_g_mass, this_g_vel):
    
    this_g_cood -= cop
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

    #this_sp_cood *= 1e3
    new = np.array(this_s_cood.dot(r))
    
    return new


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


def extract_info(num, tag, kernel='sph-anarchy', inp='GEAGLE'):
    """
    
    Args:
        num (str): the G-EAGLE id of the sim; eg: '00', '01', ...
        tag (str): the file tag; eg: '000_z015p00', '001_z014p000',...., '011_z004p770'
    
    Selects only galaxies with stellar mass > 10^8Msun inside 30pkpc
    
    """
    
    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print ("rank={}, size={}".format(rank, size))
    
    kinp = np.load('kernel_{}.npz'.format(kernel), allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']
    if rank == 0:
        print('\n\t\t\t###################################\nKernel used is `{}` and the number of bins in the look up table is {}\n\t\t\t###################################\n'.format(header.item()['kernel'], kbins))
    
    num = str(num)
    if inp == 'GEAGLE':
        if len(num) == 1:
            num =  '0'+num
        sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_{}/data'.format(num) 
    else:
        inp = 0#int(input('Enter the appropriate number for the required cosmological box: \n0 - EAGLE-REF 100 \n1 - AGN-dT9 50\n'))
        sim = ['/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/', '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'][inp]
    
    print (sim)
    z = E.read_header('SUBFIND', sim, tag, 'Redshift')
    mstar = E.read_array('SUBFIND', sim, tag, '/Subhalo/ApertureMeasurements/Mass/030kpc', numThreads=4, noH=True, physicalUnits=True)[:,4]*1e10
    sgrpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/SubGroupNumber', numThreads=4)
    grpno = E.read_array('SUBFIND', sim, tag, '/Subhalo/GroupNumber', numThreads=4)
    cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=False, physicalUnits=False, numThreads=4) #units of cMpc/h 
    #vel = E.read_array('SUBFIND', sim, tag, '/Subhalo/Velocity', noH=True, physicalUnits=True, numThreads=4)
    
    if inp == 'GEAGLE':    
        cen, r, min_dist = fit_sphere(sim, tag, 14)  #units of cMpc/h 
        #indices = np.where(np.logical_and(mstar >= 10**8, np.sqrt(np.sum((cop-cen)**2, axis = 1))<=min_dist-2.) == True)[0]
        indices = np.where(np.logical_and(mstar >= 10**7, np.sqrt(np.sum((cop-cen)**2, axis = 1))<=14) == True)[0]
    else:
        indices = np.where(mstar >= 10**7)[0]    
    
    cop = E.read_array('SUBFIND', sim, tag, '/Subhalo/CentreOfPotential', noH=True, physicalUnits=True, numThreads=4)    
    
    sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=False, physicalUnits=False, numThreads=4) 
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=False, physicalUnits=False, numThreads=4)
    if inp == 'GEAGLE': 
        dist = max(14, min_dist-2)  
        sind =  np.where(norm(sp_cood-cen,axis=1)<=dist)[0] 
        gind =  np.where(norm(gp_cood-cen,axis=1)<=dist)[0]    
    else:
        sind = np.ones(len(sp_cood), dtype=bool) 
        gind = np.ones(len(gp_cood), dtype=bool)
           
    sp_cood = E.read_array('PARTDATA', sim, tag, '/PartType4/Coordinates', noH=True, physicalUnits=True, numThreads=4)[sind]   
    gp_cood = E.read_array('PARTDATA', sim, tag, '/PartType0/Coordinates', noH=True, physicalUnits=True, numThreads=4)[gind]
    
    sp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType4/SubGroupNumber', numThreads=4)[sind]  
    sp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType4/GroupNumber', numThreads=4)[sind]  
    sp_mass = E.read_array('PARTDATA', sim, tag, '/PartType4/Mass', noH=True, physicalUnits=True, numThreads=4)[sind] * 1e10
    sp_Z = E.read_array('PARTDATA', sim, tag, '/PartType4/Metallicity', numThreads=4)[sind]  
    #sp_vel = E.read_array('PARTDATA', sim, tag, '/PartType4/Velocity', noH=True, physicalUnits=True, numThreads=4)
    sp_sl = E.read_array('PARTDATA', sim, tag, '/PartType4/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)[sind]  
    sp_ft = E.read_array('PARTDATA', sim, tag, '/PartType4/StellarFormationTime', noH=True, physicalUnits=True, numThreads=4)[sind]  

    
    gp_sgrpn = E.read_array('PARTDATA', sim, tag, '/PartType0/SubGroupNumber', numThreads=4)[gind]
    gp_grpn = E.read_array('PARTDATA', sim, tag, '/PartType0/GroupNumber', numThreads=4)[gind]
    gp_mass = E.read_array('PARTDATA', sim, tag, '/PartType0/Mass', noH=True, physicalUnits=True, numThreads=4)[gind] * 1e10
    gp_Z = E.read_array('PARTDATA', sim, tag, '/PartType0/Metallicity', numThreads=4)[gind]
    #gp_vel = E.read_array('PARTDATA', sim, tag, '/PartType0/Velocity', noH=True, physicalUnits=True, numThreads=4)
    gp_sl = E.read_array('PARTDATA', sim, tag, '/PartType0/SmoothingLength', noH=True, physicalUnits=True, numThreads=4)[gind]

    gc.collect()

    if rank == 0:
        print("Extracting required properties for {} subhalos from G-EAGLE_{} at z = {}".format(len(indices), num, z))
    
    part = int(len(indices)/size)
    comm.Barrier()

    if rank!=size-1:
        thisok = indices[rank*part:(rank+1)*part]
    else:
        thisok = indices[rank*part:]
    
    tsnum = np.zeros(len(thisok)+1).astype(int)
    tgnum = np.zeros(len(thisok)+1).astype(int)
    
    sn = len(sp_mass)
    gn = len(gp_mass)

    tsmass = np.empty(sn)
    tgmass = np.empty(gn)
    
    tsZ = np.empty(sn)
    tgZ = np.empty(gn)
    
    tscood = np.empty((sn,3))
    tgcood = np.empty((gn,3))
    
    ts_sml = np.empty(sn) 
    tg_sml = np.empty(gn) 
    
    tsage = np.empty(sn)
    tZ_los = np.empty(sn)
    
    gc.collect()
    
    ind = np.array([])    
    
    #Getting the indices (major) and the calculation of Z-LOS are the bottlenecks of this code. Maybe try 
    #cythonizing the Z-LOS part. Don't know how to change the logical operation part.
    kk = 0
    
    for ii, jj in enumerate(thisok):
        
        start = timeit.default_timer()
        s_ok = np.where(np.logical_and(sp_sgrpn-sgrpno[jj]==0,sp_grpn-grpno[jj]==0))[0]
        s_ok = s_ok[norm(sp_cood[s_ok]-cop[jj],axis=1)<=0.03]
        g_ok = np.where(np.logical_and(gp_sgrpn-sgrpno[jj]==0,gp_grpn-grpno[jj]==0))[0] 
        g_ok = g_ok[norm(gp_cood[g_ok]-cop[jj],axis=1)<=0.03]
        stop = timeit.default_timer()
        
        if len(s_ok) + len(g_ok) >= 100:
        
            print ("Calculating indices took {}s".format(np.round(stop - start,6)))
            #Use the `make_faceon` function here to transform to face-on coordinates. Need to add more
            #to make use of it. At the moment everything along the z-axis
            start = timeit.default_timer()
            Z_los_SD = get_Z_LOS(sp_cood[s_ok], gp_cood[g_ok], gp_mass[g_ok], gp_Z[g_ok], gp_sl[g_ok], lkernel, kbins)
            stop = timeit.default_timer()
            print ("Calculating Z_los took {}s".format(np.round(stop - start,6)))
            
            start = timeit.default_timer()
            tsnum[kk+1] = len(s_ok)
            tgnum[kk+1] = len(g_ok)
            
            scum = np.cumsum(tsnum)
            gcum = np.cumsum(tgnum)            
            sbeg = scum[kk]
            send = scum[kk+1]
            gbeg = gcum[kk]
            gend = gcum[kk+1]
            
            tscood[sbeg:send] = sp_cood[s_ok]
            tgcood[gbeg:gend] = gp_cood[g_ok]
            
            tsmass[sbeg:send] = sp_mass[s_ok]
            tgmass[gbeg:gend] = gp_mass[g_ok]
            
            tsZ[sbeg:send] = sp_Z[s_ok]
            tgZ[gbeg:gend] = gp_Z[g_ok]
            
            ts_sml[sbeg:send] = sp_sl[s_ok]
            tg_sml[gbeg:gend] = gp_sl[g_ok]
            
            tsage[sbeg:send] = sp_ft[s_ok]
            tZ_los[sbeg:send] = Z_los_SD
            stop = timeit.default_timer()
            print ("Assigning arrays took {}s".format(np.round(stop - start,6)))
            gc.collect()
            kk+=1
        else:
        
            ind = np.append(ind, ii)
       
    ##End of loop ii, jj##
    del sgrpno, grpno, sp_sgrpn, sp_grpn, sp_mass, sp_Z, sp_cood, sp_sl, sp_ft, gp_sgrpn, gp_grpn, gp_mass, gp_Z, gp_cood, gp_sl

    gc.collect()

    
    thisok = np.delete(thisok, ind.astype(int))    
    
    tstot = np.sum(tsnum)
    tgtot = np.sum(tgnum)
    
    tsnum = tsnum[1:len(thisok)+1]
    tgnum = tgnum[1:len(thisok)+1]
    
    tscood = tscood[:tstot]
    tgcood = tgcood[:tgtot]
    
    tsmass = tsmass[:tstot] 
    tgmass = tgmass[:tgtot] 
    
    tsZ = tsZ[:tstot] 
    tgZ = tgZ[:tgtot] 
    
    ts_sml = ts_sml[:tstot]
    tg_sml = tg_sml[:tgtot]
    
    tsage = get_age(tsage[:tstot], z, 4)
    tZ_los = tZ_los[:tstot]
    
    comm.Barrier()
    
    if rank == 0:
        
        print ("Gathering data from different processes")
    
    indices = comm.gather(thisok, root=0)
    snum = comm.gather(tsnum, root=0)
    gnum = comm.gather(tgnum, root=0)
    scood = comm.gather(tscood, root=0)
    gcood = comm.gather(tgcood, root=0)
    smass = comm.gather(tsmass, root=0)
    gmass = comm.gather(tgmass, root=0)
    sZ = comm.gather(tsZ, root=0)
    gZ = comm.gather(tgZ, root=0)
    s_sml = comm.gather(ts_sml, root=0)
    g_sml = comm.gather(tg_sml, root=0)
    sage = comm.gather(tsage, root=0)
    Z_los = comm.gather(tZ_los, root=0)
    
    
    if rank == 0:
        
        print ("Gathering completed")    
        indices = np.concatenate(np.array(indices))
        snum = np.concatenate(np.array(snum))
        gnum = np.concatenate(np.array(gnum))
        
        scood = np.concatenate(np.array(scood), axis = 0)  
        gcood = np.concatenate(np.array(gcood), axis = 0)  
        
        smass = np.concatenate(np.array(smass))
        gmass = np.concatenate(np.array(gmass))
        
        sZ = np.concatenate(np.array(sZ))
        gZ = np.concatenate(np.array(gZ))
        
        s_sml = np.concatenate(np.array(s_sml))
        g_sml = np.concatenate(np.array(g_sml))
        
        sage = np.concatenate(np.array(sage))
        Z_los = np.concatenate(np.array(Z_los))
    
   
    return mstar[indices], cop[indices], snum, gnum, scood, gcood, smass, gmass, sZ, gZ, s_sml, g_sml, sage, Z_los

##End of function `extract_info`        

def save_to_hdf5(num, tag, kernel='sph-anarchy', inp='GEAGLE'):
    
    num = str(num)
    if inp == 'GEAGLE':
        if len(num) == 1:
            num =  '0'+num
        filename = 'data2/GEAGLE_{}_sp_info.hdf5'.format(num)
    else:
        inp = 0#int(input('Enter the appropriate number for the required cosmological box: \n0 - EAGLE-REF 100 \n1 - AGN-dT9 50\n'))
        sim = ['/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data', '/cosma5/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'][inp]
        filename = 'data2/EAGLE_{}_sp_info.hdf5'.format(['REF', 'AGNdT9'][inp])
    
    ## MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
    
        print ("#################    Saving required properties to hdf5     #############")
        print ("#################   Number of processors being used is {}   #############".format(size))
    
    
    mstar, cop, snum, gnum, scood, gcood, smass, gmass, sZ, gZ, s_sml, g_sml, sage, Z_los = extract_info(num, tag, kernel, inp)

    if rank == 0:
    
        print("Wrting out required properties to hdf5")
        
        hdf5_store = HDF5_write(filename)
        hdf5_store.create_grp(tag)

        hdf5_store.create_grp('{}/Subhalo'.format(tag))
        hdf5_store.create_grp('{}/Particle'.format(tag))
        

        hdf5_store.create_dset(mstar, 'Mstar_30', '{}/Subhalo'.format(tag))
        hdf5_store.create_dset(cop.T, 'COP', '{}/Subhalo'.format(tag))

        hdf5_store.create_dset(snum, 'S_Length', '{}/Subhalo'.format(tag), dtype = np.int64)
        hdf5_store.create_dset(gnum, 'G_Length', '{}/Subhalo'.format(tag), dtype = np.int64)
        
        hdf5_store.create_dset(scood.T, 'S_Coordinates', '{}/Particle'.format(tag))
        hdf5_store.create_dset(gcood.T, 'G_Coordinates', '{}/Particle'.format(tag))
                
        hdf5_store.create_dset(smass, 'S_Mass', '{}/Particle'.format(tag))
        hdf5_store.create_dset(gmass, 'G_Mass', '{}/Particle'.format(tag))
        
        hdf5_store.create_dset(sZ, 'S_Z', '{}/Particle'.format(tag))
        hdf5_store.create_dset(gZ, 'G_Z', '{}/Particle'.format(tag))
        
        hdf5_store.create_dset(s_sml, 'S_sml', '{}/Particle'.format(tag))
        hdf5_store.create_dset(g_sml, 'G_sml', '{}/Particle'.format(tag))
        
        hdf5_store.create_dset(sage, 'S_Age', '{}/Particle'.format(tag))
        hdf5_store.create_dset(Z_los, 'S_los', '{}/Particle'.format(tag))     

