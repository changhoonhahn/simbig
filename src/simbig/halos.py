'''

module to inferace with different N-body catalogs incl. Quijote 


'''
import os
import numpy as np 
from .sims import quijote as Quijote 


def Quijote_LHC_HR(i, z=0.5): 
    ''' Read halo catalog from the high resolution Quijote LHC. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR LHC halo catalog  
    '''
    # directory that contains the Quijote LHC HR
    halo_folder = os.path.join(os.environ['QUIJOTE_DIR'],
            'Halos/latin_hypercube', 'HR_%i' % i)
    
    # look up cosmology of the LHC realization
    Om, Ob, h, ns, s8 = Quijote_LHC_cosmo(i)
    
    # read halo catalog 
    halos = Quijote.Halos(halo_folder, z, Om=Om, Ob=Ob, h=h, ns=ns, s8=s8, Mnu=0.)
    return halos


def Quijote_LHC_HR_Rockstar(i, z=0.5): 
    ''' Read Rockstar halo catalog from the high resolution Quijote at fiducial 
    cosmology. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR fiducial halo catalog  
    '''
    from . import util as UT
    import nbodykit.lab as NBlab

    # directory that contains the Quijote LHC HR
    halo_folder = os.path.join(os.environ['QUIJOTE_DIR'],
            'Halos/rockstar/latin_hypercube_HR', '%i' % i)

    # redshift snapshot 
    assert z == 0.5 
    snapnum = {0.: 4, 0.5: 3, 1.:2, 2.: 1, 3.: 0}[z]

    fhalo = os.path.join(halo_folder, 'quijote_LHC_HR%i.%i.rockstar.bf' % (i, snapnum))
    _halos = UT.read_bf_hod(fhalo) 

    halos = NBlab.HaloCatalog(
            _halos, 
            cosmo=_halos.cosmo, 
            redshift=z, 
            mdef='vir') 
    halos['Length']         = _halos['Length']
    halos['Concentration']  = _halos['Concentration']
    return halos


def Quijote_fiducial_HR(i, z=0.5): 
    ''' Read halo catalog from the high resolution Quijote at fiducial cosmology. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR fiducial halo catalog  
    '''
    # directory that contains the Quijote LHC HR
    halo_folder = os.path.join(os.environ['QUIJOTE_DIR'],
            'Halos/fiducial_HR', '%i' % i)
    
    # fiducial cosmology (Villaesuca-Navarro+2020) 
    Om, Ob, h, ns, s8 = Quijote_fiducial_cosmo() 
    
    # read halo catalog 
    halos = Quijote.Halos(halo_folder, z, Om=Om, Ob=Ob, h=h, ns=ns, s8=s8, Mnu=0.)
    return halos


def Quijote_fiducial_HR_Rockstar(i, z=0.5): 
    ''' Read Rockstar halo catalog from the high resolution Quijote at fiducial 
    cosmology. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR fiducial halo catalog  
    '''
    from . import util as UT
    import nbodykit.lab as NBlab

    # directory that contains the Quijote LHC HR
    halo_folder = os.path.join(os.environ['QUIJOTE_DIR'],
            'Halos/rockstar/fiducial_HR', '%i' % i)

    # redshift snapshot 
    snapnum = {0.: 4, 0.5: 3, 1.:2, 2.: 1, 3.: 0}[z]
    fhalo = os.path.join(halo_folder, 'quijote_fid_hr%i.%i.rockstar.bf' % (i, snapnum))
    _halos = UT.read_bf_hod(fhalo) 

    halos = NBlab.HaloCatalog(
            _halos, 
            cosmo=_halos.cosmo, 
            redshift=z, 
            mdef='vir') 
    halos['Length']         = _halos['Length']
    halos['Concentration']  = _halos['Concentration']
    return halos


def Quijote_LHC_cosmo(i): 
    ''' cosmology look up for LHC realization i at redshift z 

    outputs: Omega_m, Omega_l, h, ns, s8

    '''
    fcosmo = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat',
            'quijote_lhc_cosmo.txt')

    # Omega_m, Omega_l, h, ns, s8
    cosmo = np.loadtxt(fcosmo, unpack=True, usecols=range(5)) 

    Om = cosmo[0][i]
    Ob = cosmo[1][i]
    h  = cosmo[2][i]
    ns = cosmo[3][i]
    s8 = cosmo[4][i]

    return Om, Ob, h, ns, s8


def Quijote_fiducial_cosmo(): 
    ''' fiducial cosmology 

    '''
    # Omega_m, Omega_l, h, ns, s8
    Om = 0.3175
    Ob = 0.049
    h  = 0.6711
    ns = 0.9624
    s8 = 0.834

    return Om, Ob, h, ns, s8


def Abacus_fiducial(i, z=0.5): 
    ''' Read Abacaus halo catalog that are divided into 8 subboxes


    Parameters
    ---------- 
    i : int 
        ith subbox. (i // 8) phase, (i % 8) subbox  

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR fiducial halo catalog  
    '''
    from . import util as UT
    import nbodykit.lab as NBlab

    assert z == 0.5 

    # directory that contains the Quijote LHC HR
    halo_folder = '/tigress/chhahn/simbig/abacus/'

    fhalo = os.path.join(halo_folder, 'abacus.c000.z0.500.%i.bf' % i)
    _halos = UT.read_bf_hod(fhalo) 

    halos = NBlab.HaloCatalog(
            _halos, 
            cosmo=_halos.cosmo, 
            redshift=z, 
            mdef='vir') 
    halos['Length']         = _halos['Length']
    halos['Concentration']  = _halos['Concentration']
    return halos


def Abacus_fiducial_cosmo(): 
    ''' abacus c000 cosmology 
    '''
    # Omega_m, Omega_l, h, ns, s8
    Om = 0.315192
    Ob = 0.049302
    h  = 0.6736
    ns = 0.9649
    s8 = 0.807952
    return Om, Ob, h, ns, s8

