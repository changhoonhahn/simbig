'''

script for constructing SIMBIG HOD catalogs for Quijote LHC simulations 
with a forward model of the BOSS LOWZ SGC sample 

v1 includes assembly bias and velocity bias parameters in the HOD 

v2 broader HOD priors based on best-fit HOD parameters from cmass/nb/0_hod_priors.py

'''
import os, sys 
import numpy as np 
from simbig import halos as Halos
from simbig import galaxies as Galaxies
from simbig import forwardmodel as FM

from simbig import util as UT 
from simbig import obs as CosmoObs

np.random.seed(1)

dat_dir = '/tigress/chhahn/simbig/cmass/'


def sample_HOD(typ='train', version='v2'):
    ''' sample HOD value based on broad priors. see `cmass/nb/0_hod_priors.ipynb 
    '''
    if typ == 'train': 
        if version in ['v2', 'v3', 'v4']: 
            hod_lower_bound = np.array([12.0, 0.1, 13.0, 13.0, 0.])
            hod_upper_bound = np.array([14.0, 0.6, 15.0, 15.0, 1.5])    
            
            dhod = hod_upper_bound - hod_lower_bound
            _hod = hod_lower_bound + dhod * np.random.uniform(0., 1, size=(5))

        elif version == 'v1': 
            frange = 1.
            _hod_fid = Galaxies.thetahod_literature('reid2014_cmass')
            hod_fid = np.array([
                _hod_fid['logMmin'], 
                _hod_fid['sigma_logM'], 
                _hod_fid['logM0'],
                _hod_fid['logM1'], 
                _hod_fid['alpha']])

            dhod = 2*np.array([0.029, 0.06, 0.13, 0.06, 0.18])

            _hod = hod_fid + dhod * np.random.uniform(-0.5 * frange, 0.5 * frange, size=(5))
    elif typ == 'test': 
        frange = 0.1

        _hod_fid = Galaxies.thetahod_literature('reid2014_cmass')
        hod_fid = np.array([
            _hod_fid['logMmin'], 
            _hod_fid['sigma_logM'], 
            _hod_fid['logM0'],
            _hod_fid['logM1'], 
            _hod_fid['alpha']])

        dhod = 2*np.array([0.029, 0.06, 0.13, 0.06, 0.18])

        _hod = hod_fid + dhod * np.random.uniform(-0.5 * frange, 0.5 * frange, size=(5))
    return _hod #{'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}


def sample_theta_bias(typ='train'): 
    ''' sample assembly and velocity bias parameters
    '''
    if typ == 'train': 
        abias = np.clip(0.2 * np.random.normal(), -1., 1.) 
        conc = np.random.uniform(0.2, 2.0, size=1)
        eta_c = np.random.uniform(0., 0.7, size=1)
        eta_s = np.random.uniform(0.2, 2.0, size=1) 
    elif typ == 'test': 
        abias = np.clip(0.02 * np.random.normal(), -1., 1.) 
        conc = np.random.uniform(0.9, 1.1, size=1)
        eta_c = np.random.uniform(0., 0.1, size=1)
        eta_s = np.random.uniform(0.9, 1.1, size=1) 

    return np.array([abias, conc, eta_c, eta_s]) 


def train_thetas(version='v2'): 
    ''' generate HOD parameters by sampling the prior 
    '''
    thetas = []
    for i_lhc in range(2518): 
        # cosmological parameters for LHC realization
        theta_cosmo = Halos.Quijote_LHC_cosmo(i_lhc) # Omega_m, Omega_l, h, ns, s8

        for i_hod in range(10): 
            # HOD parameters            
            theta = sample_HOD(typ='train', version=version)
            abias, conc, eta_c, eta_s = sample_theta_bias()

            theta = np.concatenate([theta_cosmo, theta, np.array([abias, conc, eta_c, eta_s])])
            thetas.append(theta)

    thetas = np.array(thetas)

    dat_dir = '/tigress/chhahn/simbig/' # local directory on tigress
    np.savetxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.theta.dat' % version), thetas,
           header='SIMBIG CMASS-SGC parameters: Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s')
    return None 


def train_mocks(i0, i1, version='v3', overwrite=False): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    dat_dir = '/tigress/chhahn/simbig/cmass/'
    # Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s
    thetas = np.loadtxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.theta.dat' % version, skiprows=1)

    n_hod = 10 

    for i_lhc in range(i0, i1+1): 
        print('LHC %i' % i_lhc)
    
        # read in halo catalog
        halos = Halos.Quijote_LHC_HR_Rockstar(i_lhc, z=0.5)

        for i_hod in range(n_hod):
            fgal = os.path.join(dat_dir, 'train', 
                    'hod.quijote_LH%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_lhc, version, i_hod))
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.hdf5', '.dat') 
            
            restart = False 
            if os.path.isfile(fgal) and os.path.isfile(fplk) and not overwrite: 
                continue
            else:
                restart = True 
            
            # HOD parameters            
            theta = {}
            theta['logMmin']    = thetas[i_lhc*n_hod+i_hod,5]
            theta['sigma_logM'] = thetas[i_lhc*n_hod+i_hod,6]
            theta['logM0']      = thetas[i_lhc*n_hod+i_hod,7]
            theta['logM1']      = thetas[i_lhc*n_hod+i_hod,8]
            theta['alpha']      = thetas[i_lhc*n_hod+i_hod,9]

            theta['mean_occupation_centrals_assembias_param1']      = thetas[i_lhc*n_hod+i_hod,10] 
            theta['mean_occupation_satellites_assembias_param1']    = thetas[i_lhc*n_hod+i_hod,10] 

            theta['conc_gal_bias.satellites']   = thetas[i_lhc*n_hod+i_hod,11]
            theta['eta_vb.centrals']            = thetas[i_lhc*n_hod+i_hod,12]
            theta['eta_vb.satellites']          = thetas[i_lhc*n_hod+i_hod,13]

            # populate with HOD
            if i_hod == 0 or restart: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')
                hod = halos.populate(Z07AB, **theta)
            else: 
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            #gals.save(fgal)
            UT.wh5_hodcatalog(gals, fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_thetas(version='v3'): 
    ''' generate HOD parameters by sampling the prior 
    '''
    thetas = []
    for i_fid in range(100): 
        # cosmological parameters for LHC realization
        theta_cosmo = Halos.Quijote_fiducial_cosmo() # Omega_m, Omega_l, h, ns, s8

        for i_hod in range(5): 
            # HOD parameters            
            theta = sample_HOD(typ='test')
            abias, conc, eta_c, eta_s = sample_theta_bias(typ='test')

            theta = np.concatenate([theta_cosmo, theta, np.array([abias, conc, eta_c, eta_s])])
            thetas.append(theta)

    thetas = np.array(thetas)

    dat_dir = '/tigress/chhahn/simbig/' # local directory on tigress
    np.savetxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.test.theta.dat' % version), thetas,
           header='SIMBIG CMASS-SGC parameters: Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s')
    return None 


def test_mocks(i0, i1, version='v3', overwrite=False): 
    ''' construct test mocks with assembly bias and velocity bias
    '''
    # Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s
    thetas = np.loadtxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test.theta.dat' % version, skiprows=1)

    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_fid, version, i_hod))
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.hdf5', '.dat') 
            
            if os.path.isfile(fgal) and os.path.isfile(fplk) and not overwrite: 
                continue

            # HOD parameters            
            theta = {}
            theta['logMmin']    = thetas[i_fid*5+i_hod,5]
            theta['sigma_logM'] = thetas[i_fid*5+i_hod,6]
            theta['logM0']      = thetas[i_fid*5+i_hod,7]
            theta['logM1']      = thetas[i_fid*5+i_hod,8]
            theta['alpha']      = thetas[i_fid*5+i_hod,9]

            theta['mean_occupation_centrals_assembias_param1']      = thetas[i_fid*5+i_hod,10] 
            theta['mean_occupation_satellites_assembias_param1']    = thetas[i_fid*5+i_hod,10] 

            theta['conc_gal_bias.satellites']   = thetas[i_fid*5+i_hod,11]
            theta['eta_vb.centrals']            = thetas[i_fid*5+i_hod,12]
            theta['eta_vb.satellites']          = thetas[i_fid*5+i_hod,13]

            if i_hod == 0: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')

                # populate with HOD
                hod = halos.populate(Z07AB, **theta)
            else: 
                # populate with HOD
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            #gals.save(fgal)
            UT.wh5_hodcatalog(gals, fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_fof_thetas(version='v3'): 
    ''' generate HOD parameters by sampling the prior 
    '''
    thetas = []
    for i_fid in range(100): 
        # cosmological parameters for LHC realization
        theta_cosmo = Halos.Quijote_fiducial_cosmo() # Omega_m, Omega_l, h, ns, s8

        for i_hod in range(5): 
            # HOD parameters            
            theta = sample_HOD(typ='test')

            #abias, conc, eta_c, eta_s
            theta = np.concatenate([theta_cosmo, theta, np.array([0., 1., 0.2, 1.])])
            thetas.append(theta)

    thetas = np.array(thetas)

    dat_dir = '/tigress/chhahn/simbig/' # local directory on tigress
    np.savetxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.test_fof.theta.dat' % version), thetas,
           header='SIMBIG CMASS-SGC parameters: Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s')
    return None 


def test_fof_mocks(i0, i1, version='v3', overwrite=False): 
    ''' construct test mocks with FOF halo catalogs and Z07 HOD catalogs
    '''
    # Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s
    thetas = np.loadtxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test_fof.theta.dat' % version, skiprows=1)

    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Quijote_fiducial_HR(i_fid, z=0.5)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests_fof',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.%s.fof_z07.%i.hdf5' % (i_fid, version, i_hod))
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.hdf5', '.dat') 
            
            restart = False 
            if os.path.isfile(fgal) and os.path.isfile(fplk) and not overwrite: 
                continue
            else:
                restart = True 
            

            # HOD parameters            
            theta = {}
            theta['logMmin']    = thetas[i_fid*5+i_hod,5]
            theta['sigma_logM'] = thetas[i_fid*5+i_hod,6]
            theta['logM0']      = thetas[i_fid*5+i_hod,7]
            theta['logM1']      = thetas[i_fid*5+i_hod,8]
            theta['alpha']      = thetas[i_fid*5+i_hod,9]

            theta['mean_occupation_centrals_assembias_param1']      = thetas[i_fid*5+i_hod,10] 
            theta['mean_occupation_satellites_assembias_param1']    = thetas[i_fid*5+i_hod,10] 

            theta['conc_gal_bias.satellites']   = thetas[i_fid*5+i_hod,11]
            theta['eta_vb.centrals']            = thetas[i_fid*5+i_hod,12]
            theta['eta_vb.satellites']          = thetas[i_fid*5+i_hod,13]

            # populate with HOD
            if i_hod == 0 or restart: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')
                hod = halos.populate(Z07AB, **theta)
            else: 
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            #gals.save(fgal)
            UT.wh5_hodcatalog(gals, fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_abacus_thetas(version='v3'): 
    ''' generate HOD parameters by sampling the prior 
    '''
    thetas = []
    for i_fid in range(200): 
        # cosmological parameters
        theta_cosmo = Halos.Abacus_fiducial_cosmo() # Omega_m, Omega_l, h, ns, s8

        for i_hod in range(5): 
            # HOD parameters            
            theta = sample_HOD(typ='test')

            theta = np.concatenate([theta_cosmo, theta])
            thetas.append(theta)

    thetas = np.array(thetas)

    np.savetxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test_abacus.theta.dat' % version, thetas,
           header='SIMBIG CMASS-SGC parameters: Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha')
    return None 


def test_abacus_mocks(i0, i1, version='v3', overwrite=False): 
    ''' construct test mocks with abacus halo catalogs and Z07 HOD catalogs
    '''
    thetas = np.loadtxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test_abacus.theta.dat' % version, skiprows=1)

    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Abacus_fiducial(i_fid)
        
        for i_hod in range(5):
            fgal = os.path.join('/tigress/chhahn/simbig/cmass/tests_abacus/',
                    'hod.abacus.c000.z0.500.%i.%s.z07.%i.hdf5' % (i_fid, version, i_hod))
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.hdf5', '.dat') 
            
            if os.path.isfile(fgal) and os.path.isfile(fplk) and not overwrite: 
                continue

            # HOD parameters 
            theta = {}
            theta['logMmin']    = thetas[i_fid*5+i_hod,5]
            theta['sigma_logM'] = thetas[i_fid*5+i_hod,6]
            theta['logM0']      = thetas[i_fid*5+i_hod,7]
            theta['logM1']      = thetas[i_fid*5+i_hod,8]
            theta['alpha']      = thetas[i_fid*5+i_hod,9]

            if i_hod == 0: 
                _Z07 = Galaxies.Zheng07Model()
                Z07 = _Z07.to_halotools(halos.cosmo, halos.attrs['redshift'], halos.attrs['mdef'])

                # populate with HOD
                hod = halos.populate(Z07, **theta)
            else: 
                # populate with HOD
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            #gals.save(fgal)
            UT.wh5_hodcatalog(gals, fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def train_halo(i0, i1): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    for i_lhc in range(i0, i1+1): 
        print('LHC %i' % i_lhc)
    
        # read in halo catalog
        halos = Halos.Quijote_LHC_HR_Rockstar(i_lhc, z=0.5)

        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos)

        fplk = os.path.join(dat_dir, 'train', 'plk', 'plk.halo.quijote_LH%i.z0p5.dat' % i_lhc)

        # measure power spectrum 
        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos, Lbox=1000., Ngrid=360, dk=0.005)

        # save power spectrum to file 
        hdr = 'k, p0k, p2k, p4k' 
        np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_halo(i0, i1): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    for i_fid in range(i0, i1+1): 
        print('Fid %i' % i_fid)
    
        # read in halo catalog
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)

        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos)

        fplk = os.path.join(dat_dir, 'tests', 'plk', 'plk.halo.quijote_fid%i.z0p5.dat' % i_fid)

        # measure power spectrum 
        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos, Lbox=1000., Ngrid=360, dk=0.005)

        # save power spectrum to file 
        hdr = 'k, p0k, p2k, p4k' 
        np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def fix_bug(): 
    ''' script for fixing any bugs. This script is here for posterity

    * 2022/05/05: something went wrong with the hod realizations 5-10. They will be deleted.  
    '''
    # 2022/05/05 fix
    for i_lhc in range(2000): 
        print('LHC %i' % i_lhc)
        for i_hod in range(5, 10):
            fgal = os.path.join(dat_dir, 'train',
                    'hod.quijote_LH%i.z0p5.cmass_sgc.v1.%i.bf' % (i_lhc, i_hod))
            fplk = os.path.join(dat_dir, 'train', 'plk', 
                    'plk.hod.quijote_LH%i.z0p5.cmass_sgc.v1.%i.dat' % (i_lhc, i_hod))

            os.system('rm -rf %s' % fgal) 
            os.system('rm %s' % fplk)

    return None 


# ---- b0k measurements ---
def train_b0k(i0, i1, version='v3', overwrite=False): 
    ''' measure bispectrum of training data 
    '''
    dat_dir = '/tigress/chhahn/simbig/cmass/'
    # Omega_m, Omega_l, h, ns, s8, logMmin, sigma_logM, logM0, logM1, alpha, abias, conc, eta_c, eta_s
    thetas = np.loadtxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.theta.dat' % version, skiprows=1)

    n_hod = 10 

    for i_lhc in range(i0, i1+1): 
        print('LHC %i' % i_lhc)
    
        for i_hod in range(n_hod):
            fgal = os.path.join(dat_dir, 'train', 
                    'hod.quijote_LH%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_lhc, version, i_hod))
            fb0k = fgal.replace('hod.', 'b0k.hod.').replace('.hdf5', '.dat') 
            
            restart = False 
            if os.path.isfile(fb0k) and not overwrite: 
                continue
            else:
                restart = True 

            # read galaxy catalog  
            gals = UT.rh5_hodcatalog(fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k1, k2, k3, b123, q123 = CosmoObs.B0k_survey(gals, rand,
                    P0=1e4, 
                    Ngrid=360, 
                    Lbox=1800,
                    step=3, 
                    Ncut=3, 
                    Nmax=50, 
                    fft='pyfftw', 
                    dxyz=np.array([735., 15., 230.]), 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k1, k2, k3, b123, q123' 
            np.savetxt(fb0k, np.array([k1, k2, k3, b123, q123]).T, 
                    fmt='%.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_b0k(i0, i1, test_set, version='v3', overwrite=False): 
    ''' construct test mocks with assembly bias and velocity bias
    '''
    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        
        for i_hod in range(5):
            if test_set == 'TEST0': 
                fgal = os.path.join(dat_dir, 'tests',
                        'hod.quijote_fid%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_fid, version, i_hod))
            elif test_set == 'TEST1': 
                fgal = os.path.join(dat_dir, 'tests_fof',
                        'hod.quijote_fid%i.z0p5.cmass_sgc.%s.fof_z07.%i.hdf5' % 
                        (i_fid, version, i_hod))
            elif test_set == 'TEST2': 
                fgal = os.path.join(dat_dir, 'tests_abacus',
                        'hod.abacus.c000.z0.500.%i.%s.z07.%i.hdf5' % (i_fid, version, i_hod))
            else: 
                raise ValueError

            fb0k = fgal.replace('hod.', 'b0k.hod.').replace('.hdf5', '.dat') 
            
            if os.path.isfile(fb0k) and not overwrite: 
                continue
    
            # read in galaxy catalo g
            gals = UT.rh5_hodcatalog(fgal) 

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k1, k2, k3, b123, q123 = CosmoObs.B0k_survey(gals, rand,
                    P0=1e4, 
                    Ngrid=360, 
                    Lbox=1800,
                    step=3, 
                    Ncut=3, 
                    Nmax=50, 
                    fft='pyfftw', 
                    dxyz=np.array([735., 15., 230.]), 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k1, k2, k3, b123, q123' 
            np.savetxt(fb0k, np.array([k1, k2, k3, b123, q123]).T, 
                    fmt='%.5e %.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 



typ = sys.argv[1]
if typ == 'fix_bug': fix_bug()
if typ == 'theta_train': 
    version = sys.argv[2]
    train_thetas(version=version)
    exit()
if typ == 'theta_test': 
    version = sys.argv[2]
    test_thetas(version=version)
if typ == 'theta_test_fof': 
    version = sys.argv[2]
    test_fof_thetas(version=version)
    exit() 
if typ == 'theta_test_abacus': 
    version = sys.argv[2]
    test_abacus_thetas(version=version)
    exit()

i0  = int(sys.argv[2]) 
i1  = int(sys.argv[3])
version = sys.argv[4]

if typ == 'train': 
    overwrite = (sys.argv[5] == 'True') 
    train_mocks(i0, i1, version=version, overwrite=overwrite) 
elif typ == 'test': 
    overwrite = (sys.argv[5] == 'True') 
    test_mocks(i0, i1, version=version, overwrite=overwrite) 
elif typ == 'test_fof': 
    overwrite = (sys.argv[5] == 'True') 
    test_fof_mocks(i0, i1, version=version, overwrite=overwrite)
elif typ == 'test_abacus': 
    overwrite = (sys.argv[5] == 'True') 
    test_abacus_mocks(i0, i1, version=version, overwrite=overwrite)
elif typ == 'train_halo': 
    train_halo(i0, i1)
elif typ == 'test_halo': 
    test_halo(i0, i1)
elif typ == 'train_b0k': 
    overwrite = (sys.argv[5] == 'True') 
    train_b0k(i0, i1, version=version, overwrite=overwrite) 
elif typ == 'test_b0k': 
    test_set = sys.argv[5]
    overwrite = (sys.argv[6] == 'True') 
    test_b0k(i0, i1, test_set.upper(), version=version, overwrite=overwrite) 

