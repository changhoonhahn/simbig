'''


module with some utility functions


'''
import h5py 
import numpy as np 
import nbodykit.lab as NBlab
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline


def wh5_hodcatalog(hod, fhod): 
    ''' write PopulatedHaloCatalog to hdf5 file
    '''
    f = h5py.File(fhod, 'w')
    
    # save attributes 
    for k in hod.attrs.keys(): 
        if k not in ['cosmo', 'gal_types']: 
            f.attrs[k] = hod.attrs[k]

    # save cosmology detail  
    f.create_group('cosmo')
    for k in hod.attrs['cosmo'].keys(): 
        f['cosmo'].attrs[k] = hod.attrs['cosmo'][k]

    # save galaxy type detail 
    f.create_group('gal_types')
    for k in hod.attrs['gal_types'].keys(): 
        f['gal_types'].attrs[k] = hod.attrs['gal_types'][k]

    # save data columns 
    for col in hod.columns: 
        f.create_dataset(col, data=np.array(hod[col]))

    f.close() 
    return None 


def rh5_hodcatalog(fhod): 
    ''' read NBlab.ArrayCatalog from hdf5 file
    '''
    f = h5py.File(fhod, 'r')
    
    # read in columns
    hod = {}
    for k in f.keys(): 
        if k not in ['cosmo', 'gal_types']: 
            hod[k] = f[k][...]
    
    gals = NBlab.ArrayCatalog(hod)
    
    # save attributes
    for k in f.attrs.keys(): 
        gals.attrs[k] = f.attrs[k]

    gals.attrs['cosmo'] = {}
    for k in f['cosmo'].attrs.keys(): 
        gals.attrs['cosmo'][k] = f['cosmo'].attrs[k]
    
    gals.attrs['gal_types'] = {}
    for k in f['gal_types'].attrs.keys(): 
        gals.attrs['gal_types'][k] = f['gal_types'].attrs[k]

    f.close() 

    # define cosmology; caution: we don't match sigma8 here 
    cosmo = NBlab.cosmology.Planck15.clone(
            h=gals.attrs['h'], 
            Omega0_b=gals.attrs['Ob'], 
            Omega0_cdm=gals.attrs['Om'] - gals.attrs['Ob'],
            m_ncdm=None, 
            n_s=gals.attrs['ns'])

    gals.cosmo = cosmo 
    return gals 


def read_bf_hod(fhod): 
    ''' read in HOD file from BigFile
    '''
    gals = NBlab.BigFileCatalog(fhod) 

    # define cosmology; caution: we don't match sigma8 here 
    cosmo = NBlab.cosmology.Planck15.clone(
            h=gals.attrs['h'], 
            Omega0_b=gals.attrs['Ob'], 
            Omega0_cdm=gals.attrs['Om'] - gals.attrs['Ob'],
            m_ncdm=None, 
            n_s=gals.attrs['ns'])

    gals.cosmo = cosmo 
    return gals


def get_nofz(z, fsky, cosmo=None): 
    ''' calculate nbar(z) given redshift values and f_sky (sky coverage
    fraction)

    Parameters
    ----------
    z : array like
        array of redshift values 
    fsky : float 
        sky coverage fraction  
    cosmo : cosmology object 
        cosmology to calculate comoving volume of redshift bins 

    Returns
    -------
    number density at input redshifts: nbar(z) 

    Notes
    -----
    * based on nbdoykit implementation 

    '''
    # calculate nbar(z) for each galaxy 
    _, edges = scott_bin_width(z, return_bins=True)

    dig = np.searchsorted(edges, z, "right")
    N = np.bincount(dig, minlength=len(edges)+1)[1:-1]

    R_hi = cosmo.comoving_distance(edges[1:]) # Mpc/h
    R_lo = cosmo.comoving_distance(edges[:-1]) # Mpc/h

    dV = (4./3.) * np.pi * (R_hi**3 - R_lo**3) * fsky

    nofz = InterpolatedUnivariateSpline(0.5*(edges[1:] + edges[:-1]), N/dV, ext='const')
    
    return nofz


def fiducial_cosmology(): 
    ''' hardcoded fiducial cosmology. This is equivalent to the fiducial cosmology 
    of Quijote. This cosmology is meant to be used for calculating galaxy observables.

    Returns
    -------
    cosmo : nbodykit.lab.cosmology object
        cosmology object with the fiducial cosmology 
    '''
    # Om, Ob, h, ns, s8 = Halos.Quijote_fiducial_cosmo()
    Om, Ob, h, ns, s8 = 0.3175, 0.049, 0.6711, 0.9624, 0.834

    cosmo = NBlab.cosmology.Planck15.clone(
                h=h,
                Omega0_b=Ob,
                Omega0_cdm=Om - Ob,
                m_ncdm=None,
                n_s=ns)
    return cosmo
