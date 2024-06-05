'''


module for reading in Quijote data 


'''
import os
import numpy as np 
import nbodykit.lab as NBlab
# --- emanu --- 
from . import readfof 
from . import readsnap as RS


quijote_zsnap_dict = {0.: 4, 0.5: 3, 1.:2, 2.: 1, 3.: 0}


def Nbody(): 
    '''
    '''
    raise NotImplementedError
    return None 


def Halos(halo_folder, z=0.5, Om=None, Ob=None, h=None, ns=None, s8=None, Mnu=0.): 
    ''' read in Quijote halo catalog given the folder and snapshot # and store it as
    a nbodykit HaloCatalog object. The HaloCatalog object is convenient for 
    populating with galaxies and etc.


    Parameters
    ----------
    halo_folder : string
        directory that contains the halo catalogs e.g. on tiger it'd be
        something like: /projects/QUIJOTE/Halos/latin_hypercube/HR_0/

    Return 
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote halo catalog  
    '''
    # if snapshot folder is not specified 
    # then all values have to be specified in kwargs
    assert all([tt is not None for tt in [Om, Ob, z, h]]) 

    # redshift snapshot 
    assert z in quijote_zsnap_dict.keys(), 'snapshots are available at z=0, 0.5, 1, 2, 3'
    snapnum = quijote_zsnap_dict[z]

    # define cosmology; caution: we don't match sigma8 here 
    cosmo = NBlab.cosmology.Planck15.clone(
            h=h, 
            Omega0_b=Ob, 
            Omega0_cdm=Om - Ob,
            m_ncdm=[None, Mnu][Mnu > 0.], 
            n_s=ns) 
    Ol = 1. - Om 
    Hz = 100.0 * np.sqrt(Om * (1. + z)**3 + Ol) # km/s/(Mpc/h)

    # read FOF catalog (~90.6 ms) 
    Fof = readfof.FoF_catalog(halo_folder, snapnum, read_IDs=False,
            long_ids=False, swap=False, SFR=False)
    group_data = {}  
    group_data['Length']    = Fof.GroupLen
    group_data['Position']  = Fof.GroupPos/1e3
    group_data['Velocity']  = Fof.GroupVel * (1 + z) # km/s
    group_data['Mass']      = Fof.GroupMass*1e10
    # calculate velocity offset
    rsd_factor = (1. + z) / Hz
    group_data['VelocityOffset'] = group_data['Velocity'] * rsd_factor
    # save to ArryCatalog for consistency
    cat = NBlab.ArrayCatalog(group_data, BoxSize=np.array([1000., 1000., 1000.])) 
    cat = NBlab.HaloCatalog(cat, cosmo=cosmo, redshift=z, mdef='vir') 
    cat['Length'] = group_data['Length']

    cat.attrs['Om'] = Om
    cat.attrs['Ob'] = Ob
    cat.attrs['Ol'] = Ol
    cat.attrs['h'] = h 
    cat.attrs['ns'] = ns
    cat.attrs['s8'] = s8
    cat.attrs['Hz'] = Hz # km/s/(Mpc/h)
    cat.attrs['rsd_factor'] = rsd_factor 
    return cat
