'''

module for measuring the observable of a galaxy catalog. Compile all the
different methods for measuring observables here.  


'''
import numpy as np
import scipy
from . import util as UT 
# --- nbodykit --- 
import nbodykit.lab as nblab 
from nbodykit.algorithms.fftpower import FFTPower
from nbodykit.source.mesh.field import FieldMesh


def Plk_survey(galaxies, randoms, weights=None, cosmo=None, Ngrid=360, dk=0.005, P0=1e4, silent=True):
    ''' Measure galaxy powerspectrum multipoles for a survey geometry using the
    `nbodykit`. This function uses the FKP estmiator for calculating the power 
    spectrum.
   

    Parameters
    ----------
    galaxies : GalaxyCatalog object

    randoms : nbodykit.ArrayCatalog object 

    weights : array_like, optional 
        weights for the galaxies

    cosmo : nobodykit.lab.cosmology object
        cosmology used to convert (RA, Dec, z) to cartesian coordinates and calculate
        nbar(z). This should be some fiducial cosmology. 

    Ngrid : int
        grid size for FFT 

    P0 : float 
        P0 value for FKP weights. (default: 1e4) 

    silent : boolean
        If True, the function will print out a bunch of stuff for sanity checks.

    
    Return
    ------
    k, p0k, p2k, p4k
        The power spectrum monopole, quadrupole, and hexadecapole 

    Notes
    -----
    * 03/09/2022: cosmology input added. previously we used the cosmology of the 
        galaxy catalog. However, in practice, we don't know the true cosmology and
        the power spectra are measured assuming some fiducial cosmology. 
    * 05/21/2021: tested; nbar(z) calculation modified
    * 04/02/2021: implemented but not yet tested
    '''
    Ng = len(galaxies) # number of galaxies 
    Nr = len(randoms) # number of randoms
    
    # weights 
    if weights is None: 
        w_g = np.ones(Ng) 
    else: 
        w_g = weights
    w_r = np.ones(Nr) 
    if not silent: print('alpha = %f' % (np.sum(w_g)/np.sum(w_r)))

    # cosmology
    if cosmo is None: 
        cosmo = UT.fiducial_cosmology()
        if not silent: print("assuming fiducial cosmology") 

    # get nbar(z) for the galaxy and random samples
    ng_of_z = UT.get_nofz(np.array(galaxies['Z']), galaxies.attrs['fsky'], cosmo=cosmo)
    nbar_g = ng_of_z(np.array(galaxies['Z']))
    nbar_r = ng_of_z(np.array(randoms['Z']))

    # calculate xyz positions
    pos_g = nblab.transform.SkyToCartesian(
            galaxies['RA'], galaxies['DEC'], galaxies['Z'], cosmo=cosmo) 
    pos_r = nblab.transform.SkyToCartesian( 
            randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo) 
    
    _gals = nblab.ArrayCatalog({
        'Position': pos_g, 
        'NZ': nbar_g, 
        'WEIGHT': w_g, 
        'WEIGHT_FKP': 1./(1. + nbar_g * P0)
        })

    _rands = nblab.ArrayCatalog({ 
        'Position': pos_r, 
        'NZ': nbar_r,
        'WEIGHT': w_r,
        'WEIGHT_FKP': 1./(1. + nbar_r * P0)
    })

    fkp = nblab.FKPCatalog(_gals, _rands)
    mesh = fkp.to_mesh(Nmesh=Ngrid, nbar='NZ', fkp_weight='WEIGHT_FKP', comp_weight='WEIGHT', window='tsc')

    # compute the multipoles
    r = nblab.ConvolvedFFTPower(mesh, poles=[0,2,4], dk=dk, kmin=0.)
    
    k = r.poles['k'] 
    p0k = r.poles['power_0'].real - r.attrs['shotnoise']
    p2k = r.poles['power_2'].real 
    p4k = r.poles['power_4'].real 
    if not silent: 
        for key in r.attrs: print("   %s = %s" % (key, str(r.attrs[key])))

    return k, p0k, p2k, p4k


def Plk_box(galaxies, Lbox=1000., Ngrid=360, dk=0.005, LOS=[0,0,1]):
    ''' Measure galaxy powerspectrum multipoles for a galaxy sample in box using 
    `nbodykit`.

    Parameters
    ----------
    galaxies : GalaxyCatalog object

    
    Return
    ------
    k, p0k, p2k, p4k
        The power spectrum monopole, quadrupole, and hexadecapole 

    
    Notes 
    -----
    * CHH: I modified the code a bit based on https://nbodykit.readthedocs.io/en/latest/cookbook/fftpower.html

    '''
    # paint galaxies to mesh
    mesh = galaxies.to_mesh(window='tsc', Nmesh=Ngrid, BoxSize=Lbox, 
            compensated=True, position='Position')

    #compute the power spectrum moments using nbodykit 
    pk_moms = FFTPower(mesh, mode='2d', dk=dk, kmin=0., poles=[0,2,4], los=LOS)
    
    k = pk_moms.poles['k'] 
    # apply shot noise correction 
    p0k =  pk_moms.poles['power_0'].real - pk_moms.attrs['shotnoise']
    p2k =  pk_moms.poles['power_2'].real
    p4k =  pk_moms.poles['power_4'].real

    return k, p0k, p2k, p4k 


def B0k_survey(galaxies, randoms, weights=None, cosmo=None, P0=1e4, Ngrid=360, Lbox=1400, step=3, Ncut=3, Nmax=40, fft='pyfftw', dxyz=None, silent=True):
    ''' Measure the bispectrum monopole for a survey geometry using 
    the `pySpectrum` package.`pySpectrum` uses the Scoccimarro (2015)
    estimator for the bispectrum.
   

    Parameters
    ----------
    galaxies : GalaxyCatalog object
    
    randoms : nbodykit.ArrayCatalog object 

    weights : array_like, optional 
        weights for the galaxies
    
    P0 : float 
        P0 value for FKP weights. (default: 1e4) 

    Ngrid : int
        grid size for FFT 
    
    Lbox : float, optional 
        box size (default: 2600.)

    Nmax : int, optional 
        number of steps to include --- i.e. number of modes. (default: 40) 
    
    Ncut : int, optional 
        k minimum in units of fundamental mode (default: 3)

    step : int, optional 
        step size in units of fundamental mode (defualt: 3) 
    
    fft : string, optional
        specifies which fftw version to use. Options are 'pyfftw' and
        'fortran'. (default: 'pyfftw') 

    dxyz : optional
        shift galaxy positions to center the box more efficiently

    silent : boolean
        If True, the function will print out a bunch of stuff for sanity checks.
    
    Return
    ------
    k1, k2, k3, b123, q123
        the bispectrum B(k1, k2, k3) and reduced bispectrum Q(k1, k2, k3) 
    
    Notes
    -----
    * 05/27/2021: CH implemented 
    '''
    # import pyspectrum (see https://github.com/changhoonhahn/pySpectrum for details) 
    from pyspectrum import pyspectrum as pySpec

    # cosmology
    if cosmo is None: 
        cosmo = UT.fiducial_cosmology()
        if not silent: print("assuming fiducial cosmology") 

    Ng = len(galaxies) # number of galaxies 
    Nr = len(randoms) # number of randoms
    
    # weights 
    if weights is None: 
        w_g = np.ones(Ng) 
    else: 
        w_g = weights
    w_r = np.ones(Nr) 
    if not silent: print('alpha = %f' % (np.sum(w_g)/np.sum(w_r)))

    # get nbar(z) for the galaxy and random samples
    ng_of_z = UT.get_nofz(np.array(galaxies['Z']), galaxies.attrs['fsky'], cosmo=cosmo)
    nbar_g = ng_of_z(np.array(galaxies['Z']))
    nbar_r = ng_of_z(np.array(randoms['Z']))

    # (RA, DEC, Z) for galaxies and random
    radecz_g = np.array([galaxies['RA'], galaxies['DEC'], galaxies['Z']]) 
    radecz_r = np.array([randoms['RA'], randoms['DEC'], randoms['Z']]) 

    # calculate bispectrum 
    bisp = pySpec.B0_survey(
            radecz_g, nbar_g, w=w_g,                        # galaxies
            radecz_r=radecz_r, nbar_r=nbar_r, w_r=w_r,      # randoms 
            P0_fkp=P0, 
            Ngrid=Ngrid,
            Lbox=Lbox, 
            step=step,
            Ncut=Ncut, 
            Nmax=Nmax, 
            cosmo=cosmo,
            fft=fft, 
            dxyz=dxyz, 
            nthreads=1, 
            precision='single', 
            silent=silent)

    k1 = bisp['meta']['kf'] * bisp['i_k1']
    k2 = bisp['meta']['kf'] * bisp['i_k2']
    k3 = bisp['meta']['kf'] * bisp['i_k3']

    b123 = bisp['b123']
    q123 = bisp['q123']

    return k1, k2, k3, b123, q123


def B0k_box(galaxies, Lbox=1400., Ngrid=360, step=3, Ncut=3, Nmax=40, fft='pyfftw', silent=True):
    ''' Measure galaxy bispectrum monopole for a periodic box using `pySpectrum`.

    Parameters
    ----------
    galaxies : GalaxyCatalog object
        galaxies in a periodic box. e.g. output from `simbig.galaxies.hodGalaxies`

    Ngrid : int
        grid size for FFT 
    
    Lbox : float, optional 
        box size (default: 2600.)

    Nmax : int, optional 
        number of steps to include --- i.e. number of modes. (default: 40) 
    
    Ncut : int, optional 
        k minimum in units of fundamental mode (default: 3)

    step : int, optional 
        step size in units of fundamental mode (defualt: 3) 
    
    fft : string, optional
        specifies which fftw version to use. Options are 'pyfftw' and
        'fortran'. (default: 'pyfftw') 

    silent : boolean
        If True, the function will print out a bunch of stuff for sanity checks.
    
    Return
    ------
    k1, k2, k3, b123, q123
        the bispectrum B(k1, k2, k3) and reduced bispectrum Q(k1, k2, k3) 
    
    Notes 
    -----
    * 05/27/2021: CHH, implemented function 
    '''
    # import pyspectrum (see https://github.com/changhoonhahn/pySpectrum for details) 
    from pyspectrum import pyspectrum as pySpec

    # x,y,z position of galaxies
    xyz = np.array(galaxies['Position']).T

    bisp = pySpec.Bk_periodic(xyz, Lbox=Lbox, Ngrid=Ngrid, step=step, Ncut=Ncut, Nmax=Nmax, fft=fft, nthreads=1, silent=silent)
    
    k1 = bisp['meta']['kf'] * bisp['i_k1']
    k2 = bisp['meta']['kf'] * bisp['i_k2']
    k3 = bisp['meta']['kf'] * bisp['i_k3']

    b123 = bisp['b123']
    q123 = bisp['q123']

    return k1, k2, k3, b123, q123


def mark_model(delta_R,delta_s,p):
    mark_m = ((delta_s+1.)/(delta_s+delta_R+1.0+1e-6))**p
    return mark_m


def Mlk_survey(radius, delta_s, p, gals, rand, weights=None, cosmo=None, Ngrid=360, dk=0.005, silent=True):
    ''' Measure the marked power spectrum multipoles for a survey geometry using                              
    `nbodykit`. This function uses the FKP estmiator for calculating the power                                
    spectrum.                                                                                                 
                                                                                                              
    Parameters                                                                                                
    ----------                                                                                                
    radius : float, scale used to compute the local density                                                   
                                                                                                              
    delta_s : float, mark parameter                                                                           
                                                                                                              
    p : float, mark parameter                                                                                 
                                                                                                              
    gals : GalaxyCatalog object                                                                               
                                                                                                              
    rand : nbodykit.ArrayCatalog object                                                                       
                                                                                                              
    weights : array_like, optional                                                                            
        weights for the galaxies                                                                              
                                                                                                              
    cosmo : nobodykit.lab.cosmology object                                                                    
        cosmology used to convert (RA, Dec, z) to cartesian coordinates and calculate                         
        nbar(z). This should be some fiducial cosmology.                                                      
                                                                                                              
    Ngrid : int 
        grid size for FFT                                                                                     
                                                                                                              
    silent : boolean                                                                                          
        If True, the function will print out a bunch of stuff for sanity checks.                              
                                                                                                              
    Return                                                                                                    
    ------                                                                                                    
    k, p0k, p2k, p4k                                                                                          
        The marked power spectrum monopole, quadrupole, and hexadecapole   
                                                                                                              
                                                                                                              
                                                                                                              
    Notes                                                                                                     
    -----                                                                                                     
    * 03/30/2022: EM implemented                                                                              
    '''


    Ng = len(gals)
    Nr = len(rand)

    # weights                                                                                                 
    if weights is None:
        w_g = np.ones(Ng)
    else:
        w_g = weights
    w_r = np.ones(Nr)
    alpha = np.sum(w_g)*1.0/np.sum(w_r)
    if not silent: print('alpha = %f' % (alpha))

    # cosmology                                                                                               
    if cosmo is None:
        cosmo = UT.fiducial_cosmology()
        if not silent: print("assuming fiducial cosmology")

    # get nbar(z) for the galaxy and random samples                                                           
    ng_of_z = UT.get_nofz(np.array(gals['Z']), gals.attrs['fsky'], cosmo=cosmo)
    nbar_g = ng_of_z(np.array(gals['Z']))
    nbar_r = ng_of_z(np.array(rand['Z']))

    # calculate xyz positions                                                                                 
    pos_g = nblab.transform.SkyToCartesian(gals['RA'], gals['DEC'], gals['Z'], cosmo=cosmo)
    pos_r = nblab.transform.SkyToCartesian(rand['RA'], rand['DEC'], rand['Z'], cosmo=cosmo)

    # compute the local density                                                                               
    tree_gals = scipy.spatial.cKDTree(pos_g)
    tree_rand = scipy.spatial.cKDTree(pos_r)
    inds_gals = tree_gals.query_ball_point(pos_g, radius)
    inds_rand = tree_rand.query_ball_point(pos_g, radius)
    delta_g = np.empty(w_g.shape[0])
    for i, ig, ir in zip(range(w_g.shape[0]), inds_gals, inds_rand):
        delta_g[i] = np.sum(w_g[ig]) / alpha / np.sum(w_r[ir])-1.0

    # compute the marked field                                                                                
    mark = mark_model(delta_g,delta_s,p)
    mark_bar = np.sum(mark*np.array(gals['Weight']))/np.sum(np.array(gals['Weight']))

    _gals = nblab.ArrayCatalog({'Position': pos_g,
                                'NZ': nbar_g*mark_bar,
                                'WEIGHT': w_g*mark,
                                'WEIGHT_FKP': np.ones(Ng) #1./(1. + nbar_g * P0)                              
                            })

    _rands = nblab.ArrayCatalog({'Position': pos_r,
                                 'NZ': nbar_r*mark_bar,
                                 'WEIGHT': w_r*mark_bar,
                                'WEIGHT_FKP': np.ones(Nr) #1./(1. + nbar_r *P0)                               
                             })

    # compute Power spectrum                                                                                  
    fkp = nblab.FKPCatalog(_gals, _rands)
    mesh = fkp.to_mesh(Nmesh=Ngrid, nbar='NZ', fkp_weight='WEIGHT_FKP',
                       comp_weight='WEIGHT', window='tsc')
    # compute the multipoles                                                                                  
    r = nblab.ConvolvedFFTPower(mesh, poles=[0,2,4], dk=0.005, kmin=0.)
    k = r.poles['k']
    p0k = r.poles['power_0'].real #- r.attrs['shotnoise']                                                     
    p2k = r.poles['power_2'].real
    p4k = r.poles['power_4'].real

    if not silent:
        for key in r.attrs: print("   %s = %s" % (key, str(r.attrs[key])))

    return k, p0k, p2k, p4k
