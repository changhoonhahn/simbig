''' 


module for forward modeling the BOSS survey: i.e. python version of
mksample 


'''
import os, time
import h5py
import numpy as np 
from .remap import Cuboid 
import nbodykit.lab as NBlab

import pymangle 
from pydl.pydlutils.spheregroup import spherematch


def additional_selection(ra, dec, z, sample='lowz-south'):
    ''' any additioanl selections that are applied to the forward model and the data 
    '''
    if sample == 'lowz-south': 
        zmin, zmax = 0.2, 0.37 
        in_select = (z > zmin) & (z < zmax) 
    elif sample == 'cmass-south': 
        zmin, zmax = 0.45, 0.6
        in_select = (z > zmin) & (z < zmax) & ((ra < 28) | (ra > 335)) & (dec > -6)
    else: 
        raise NotImplementedError
    return in_select


def BOSS(_galaxies, sample='lowz-south', seed=0, veto=True, fiber_collision=True, silent=True):
    ''' Forward model the BOSS survey given a simulated galaxy catalog 
    '''
    galaxies = _galaxies.copy()
    assert sample in ['lowz-south', 'cmass-south'], 'only LOWZ and CMASS SGC has been implemented' 
    assert np.all(galaxies.attrs['BoxSize'] == 1000.), 'only supported for 1Gpc/h cubic box'
   
    # use BoxRemap to transform the volume (https://arxiv.org/abs/1003.3178)
    # at the moment this takes about ~5sec --- but it can definitely be sped up.
    # (check code pacakge for list of boxremap examples) 
    if sample == 'lowz-south': 
        C = Cuboid(u1=(1,1,0), u2=(0,1,0), u3=(0,0,1))
    elif sample == 'cmass-south': 
        C = Cuboid(u1=(1,1,0), u2=(1,0,1), u3=(1,0,0))
        
    xyz = np.array(galaxies['Position']) / 1000.
    xyz_t = np.empty(xyz.shape)
    for i in range(xyz.shape[0]): 
        try: 
            xyz_t[i,:] = C.Transform(xyz[i,0], xyz[i,1], xyz[i,2]) # transformed
        except RuntimeError:
            try: 
                xyz_t[i,:] = C.Transform(xyz[i,0].astype(np.float32), xyz[i,1].astype(np.float32), xyz[i,2].astype(np.float32))
            except RuntimeError:
                xyz_t[i,:] = C.Transform(xyz[i,0].astype(np.float16), xyz[i,1].astype(np.float16), xyz[i,2].astype(np.float16))
    xyz_t *= 1000. 
        
    vxyz = np.array(galaxies['Velocity'])
    vxyz_t = np.empty(vxyz.shape) 
    for i in range(vxyz.shape[0]): 
        vxyz_t[i,:] = C.TransformVelocity(vxyz[i,0], vxyz[i,1], vxyz[i,2]) # transformed
        
    # rotate BoxRemap-ed cuboid 
    if sample == 'lowz-south': 
        xyz_t   = np.dot(xyz_t, np.array([[0, -1, 0], [1, 0, 0,], [0, 0, 1]])) 
        vxyz_t  = np.dot(vxyz_t, np.array([[0, -1, 0], [1, 0, 0,], [0, 0, 1]])) 
    elif sample == 'cmass-south': 
        xyz_t   = np.dot(xyz_t, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
        vxyz_t  = np.dot(vxyz_t, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])) 

    # translate 
    if sample == 'lowz-south': 
        xyz_t += np.array([334.45, 738.4, -351.1])[None,:] # translate 
    elif sample == 'cmass-south': 
        # translate and rotate to fit box more optimally
        xyz_t += np.array([500 * C.L2, -500 * C.L1, -500 * C.L3 + 1291])[None,:]
        
        # these are tuned in the notebook `nb/fit_cmass.ipynb`
        theta_y = -1.3
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        theta_z = -0.0294923026047
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

        xyz_t   = np.dot(xyz_t, Ry)
        vxyz_t  = np.dot(vxyz_t, Ry)

        xyz_t   = np.dot(xyz_t, Rz)
        vxyz_t  = np.dot(vxyz_t, Rz)
    
    # transform Cartesian to (RA, Dec, z) 
    ra, dec, z = NBlab.transform.CartesianToSky(
            xyz_t, 
            galaxies.cosmo,
            velocity=vxyz_t, 
            observer=[0,0,0])
    galaxies['RA']  = ra
    galaxies['DEC'] = dec 
    galaxies['Z']   = z 

    # angular mask
    if not silent: t0 = time.time() 
    boss_poly = BOSS_mask(sample)
    in_footprint = BOSS_angular(ra, dec, mask=boss_poly)
    if not silent: print('..applying angular mask takes %.f sec' % (time.time() - t0))

    # veto mask 
    if veto: 
        if not silent: t0 = time.time() 
        in_veto = BOSS_veto(ra, dec) 
        if not silent: print('..applying veto takes %.f sec' % (time.time() - t0))
        in_footprint = in_footprint & ~in_veto
    
    # further selection functions
    if not silent: print('..applying additional selection')
    in_select = additional_selection(ra, dec, z, sample=sample) 

    #if not silent: t0 = time.time() 
    #in_nz = BOSS_radial(z[in_footprint], sample=sample, seed=seed)
    #in_select = np.zeros(len(ra)).astype(bool) 
    #in_select[np.arange(len(ra))[in_footprint][in_nz]] = True
    #if not silent: print('..applying raidal takes %.f sec' % (time.time() - t0))

    select = in_footprint & in_select

    if fiber_collision: # apply fiber collisions
        if not silent: t0 = time.time() 
        _fibcoll = BOSS_fibercollision(np.array(ra)[select], np.array(dec)[select])

        fibcoll = np.zeros(len(ra)).astype(bool) 
        fibcoll[np.arange(len(ra))[select][_fibcoll]] = True

        if not silent: print('..applying fiber collisions takes %.f sec' % (time.time() - t0))
    else: 
        fibcoll = np.zeros(len(ra)).astype(bool) 

    galaxies = galaxies[select & ~fibcoll]
    
    area = BOSS_area(sample=sample, veto=veto)
    fsky = area / (360.**2 / np.pi)
    if not silent: print("..footprint covers %.3f of sky" % fsky)
    galaxies.attrs['fsky'] = fsky 
    return galaxies


def BOSS_mask(sample): 
    ''' read mangle polygon for specified sample 
    '''
    if sample == 'lowz-south': 
        f_poly = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                'dat', 'mask_DR12v5_LOWZ_South.ply') 
    elif sample == 'cmass-south': 
        f_poly = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                'dat', 'mask_DR12v5_CMASS_South.ply')
    else: 
        raise NotImplementedError
    boss_poly = pymangle.Mangle(f_poly) 
    return boss_poly


def BOSS_angular(ra, dec, mask=None): 
    ''' Given RA and Dec, check whether the galaxies are within the angular
    mask of BOSS
    '''
    w = mask.weight(ra, dec)
    inpoly = (w > 0.) 
    return inpoly 


def BOSS_veto(ra, dec): 
    ''' given RA and Dec, find the objects that fall within one of the veto 
    masks of BOSS. At the moment it checks through the veto masks one by one.  
    '''
    in_veto = np.zeros(len(ra)).astype(bool) 
    fvetos = [
            'badfield_mask_postprocess_pixs8.ply', 
            'badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply',
            'allsky_bright_star_mask_pix.ply',
            'bright_object_mask_rykoff_pix.ply', 
            'centerpost_mask_dr12.ply', 
            'collision_priority_mask_dr12.ply']

    veto_dir = os.path.join(os.environ['QUIJOTE_DIR'], 'chang', 'simbig')
    for fveto in fvetos: 
        veto = pymangle.Mangle(os.path.join(veto_dir, fveto))
        w_veto = veto.weight(ra, dec)
        in_veto = in_veto | (w_veto > 0.)
    return in_veto


def BOSS_fibercollision(ra, dec): 
    ''' apply BOSS fiber collisions 
    '''
    fib_angscale = 0.01722 # 62'' fiber collision angular scale 
    m1, m2, d12 = spherematch(ra, dec, ra, dec, fib_angscale, maxmatch=2) 

    notitself = (d12 > 0.0) 

    m1 = m1[notitself]
    m2 = m2[notitself]
    
    # pairs are double counted by spherematch. This selects the unique pairs 
    _, ipair = np.unique(np.min(np.array([m1, m2]), axis=0), return_index=True)

    # only ~60% of galaxies within the angular scale are fiber collided 
    # since 40% are in overlapping regions with substantially lower 
    # fiber collision rates 
    ncollid = int(0.6 * len(ipair))
    
    icollid = np.random.choice(ipair, size=ncollid, replace=False) 
    
    fibcollided = np.zeros(len(ra)).astype(bool)
    fibcollided[m1[icollid[:int(0.5*ncollid)]]] = True
    fibcollided[m2[icollid[int(0.5*ncollid):]]] = True
    return fibcollided 


def BOSS_radial(z, sample='lowz-south', seed=0): 
    ''' Downsample the redshifts to match the BOSS radial selection function.
    This assumes that the sample consists of the same type of galaxies (i.e. 
    constant HOD), but selection effects randomly remove some of them 

    Notes
    -----
    * nbar file from https://data.sdss.org/sas/bosswork/boss/lss/DR12v5/
    '''
    if sample == 'lowz-south': 
        f_nbar = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                    'dat', 'nbar_DR12v5_LOWZ_South_om0p31_Pfkp10000.dat') 
        zmin, zmax = 0.2, 0.37 
    else: 
        raise NotImplementedError

    # zcen,zlow,zhigh,nbar,wfkp,shell_vol,total weighted gals
    zcen, zlow, zhigh, nbar, wfkp, shell_vol, tot_gal = np.loadtxt(f_nbar, 
            skiprows=2, unpack=True) 
    zedges = np.concatenate([zlow, [zhigh[-1]]])

    ngal_z, _ = np.histogram(np.array(z), bins=zedges)

    # fraction to downsample
    fdown_z = tot_gal/ngal_z.astype(float)

    # impose redshift limit 
    zlim = (z > zmin) & (z < zmax) 

    i_z = np.digitize(z, zedges)
    downsample = (np.random.rand(len(z)) < fdown_z[i_z])

    return zlim #& downsample 


def BOSS_area(sample='lowz-south', veto=True): 
    ''' get footprint area of BOSS sample. Currently everything is hardcoded


    Returns
    -------
    footprint of boss sample in sq. degrees


    Notes
    -----
    * veto fraction is hardcoded using values from https://data.sdss.org/sas/dr12/boss/lss/mksampleDR12/nbarutils.py

    '''
    if sample == 'lowz-south': 
        f_poly = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                'dat', 'mask_DR12v5_LOWZ_South.ply') 
    elif sample == 'cmass-south': 
        f_poly = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                'dat', 'mask_DR12v5_CMASS_South.ply') 
    else: 
        raise NotImplementedError
    boss_poly = pymangle.Mangle(f_poly) 

    area = np.sum(boss_poly.areas * boss_poly.weights) # deg^2

    if sample == 'cmass-south': 
        area *= 0.707104 # estimated using ransack

        if veto: area *= 0.8983456464678463
    else: 
        if veto: # remove veto mask areas 
            fveto = 0.9012 # estimated from mangle ransack 

            #fveto = 0.5*(0.906614+0.906454) # from https://data.sdss.org/sas/dr12/boss/lss/mksampleDR12/nbarutils.py
            '''
            fvetos = [
                    'badfield_mask_postprocess_pixs8.ply', 
                    'badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply',
                    'allsky_bright_star_mask_pix.ply',
                    'bright_object_mask_rykoff_pix.ply', 
                    'centerpost_mask_dr12.ply', 
                    'collision_priority_mask_dr12.ply']
            for fveto in fvetos: 
                veto = pymangle.Mangle(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', fveto))
                area -= veto.area 
            '''
            area *= fveto 
    return area 


def BOSS_randoms(boss_gals, weights=None, sample='lowz-south', veto=False): 
    ''' given forward modeled galaxy catalog (output from the BOSS function) construct 
    accompanying random catalog. 

    Parameters 
    ----------
    boss_gals : catalog object
        catalog output from `forwardmodel.BOSS` function 
    weights : array_like
        corresponding weights for the galaxy catalog
    sample : string
        specify which BOSS sample. currently only supports 'lowz-south'
    veto : boolean
        if True, veto mask is applied on the random 

    Returns
    -------
    nbodykit.ArrayCatalog with RA, DEC, and Z of random catalog

    '''
    from astropy.stats import scott_bin_width
    if sample not in ['lowz-south', 'cmass-south']: 
        raise NotImplementedError('%s not yet supported' % sample)

    if sample == 'lowz-south': 
        frand = os.path.join(os.environ['QUIJOTE_DIR'], 'chang', 
                'random_DR12v5_LOWZ_South.hdf5')
    elif sample == 'cmass-south': 
        frand = os.path.join(os.environ['QUIJOTE_DIR'], 'chang', 
                'random_DR12v5_CMASS_South.hdf5')

    if veto: # with veto mask 
        frand = frand.replace('.hdf5', '.veto.hdf5')
    
    # read RA and Dec values 
    rand = h5py.File(frand, 'r') 
    rand_ra     = rand['ra'][...]
    rand_dec    = rand['dec'][...]

    # generate redshifts that match input galaxy redshift distribution. This 
    # implementation is similiar to the implementation in nbodykit 
    w, bins = scott_bin_width(np.array(boss_gals['Z']), return_bins=True)
    hist, edges = np.histogram(np.array(boss_gals['Z']), bins=bins, weights=weights) 
    cutoffs = np.cumsum(hist) / np.sum(hist)

    prng = np.random.uniform(size=len(rand_ra))
    rand_z = edges[:-1][cutoffs.searchsorted(prng)] + w * np.random.uniform(size=len(rand_ra))

    # further selection functions
    in_select = additional_selection(rand_ra, rand_dec, rand_z, sample=sample)

    return NBlab.ArrayCatalog({'RA': rand_ra[in_select], 'DEC': rand_dec[in_select], 'Z': rand_z[in_select]})


def Box_RSD(cat, LOS=[0,0,1], Lbox=1000.):
    ''' Given a halo/galaxy catalog in a periodic box, apply redshift space 
    distortion specified LOS along LOS

    Parameters
    ----------
    cat : CatalogBase
        nbodykit.Catalog object
    LOS : array_like 
        3 element list specifying the direction of the line-of-sight
    Lbox : float
        box size in Mpc/h
    '''
    pos = np.array(cat['Position']) + np.array(cat['VelocityOffset']) * LOS

    # impose periodic boundary conditions for particles outside the box
    i_rsd = np.arange(3)[np.array(LOS).astype(bool)][0]
    rsd_pos = pos[:,i_rsd] % Lbox
    pos[:,i_rsd] = np.array(rsd_pos)
    return pos
