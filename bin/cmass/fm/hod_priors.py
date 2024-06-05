'''


script to determine HOD priors 


'''
import os, sys
import numpy as np
from scipy.optimize import optimize
# simbig
from simbig import halos as Halos
from simbig import galaxies as Galaxies
from simbig import obs as CosmoObs


run = sys.argv[1]


# LHC cosmologies 
theta_cosmos = np.loadtxt('/home/chhahn/projects/simbig/src/simbig/dat/quijote_lhc_cosmo.txt', unpack=True, usecols=range(5))

# observed P(k) 
dat_dir = '/tigress/chhahn/simbig/' 
k_obs = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.k.dat'), skiprows=1)
p0k_obs = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p0k.w_nofc.dat'), skiprows=1)
p2k_obs = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p2k.w_nofc.dat'), skiprows=1)
p4k_obs = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p4k.w_nofc.dat'), skiprows=1)


def high_Om(): 
    ''' get best-fit HOD for high Om
    '''
    # pick random high Om 
    i_high_Om = np.random.choice(np.arange(theta_cosmos.shape[0])[theta_cosmos[:,0] > 0.495])
    print(theta_cosmos[i_high_Om])

    # import halos 
    halos_high_Om = halos = Halos.Quijote_LHC_HR_Rockstar(i_high_Om, z=0.5)

    def Pk_highOm(logMmin, sigma_logM, logM0, logM1, alpha):
        hod = {}
        hod['logMmin'] = logMmin
        hod['sigma_logM'] = sigma_logM
        hod['logM0'] = logM0
        hod['logM1'] = logM1
        hod['alpha'] = alpha

        gal = Galaxies.hodGalaxies(halos_high_Om, hod, seed=0)
        _k, _p0k, _p2k, _p4k = CosmoObs.Plk_box(gal)
        return _k[_k < 0.5], _p0k[_k < 0.5], _p2k[_k < 0.5], _p4k[_k < 0.5]


    def chi2_P0k_highOm(tt):
        logMmin, sigma_logM, logM0, logM1, alpha = tt

        if sigma_logM < 1e-4:
            return 1e12
        if logMmin < 12 or logM0 < 12 or logM1 < 12: 
            return 1e12 
        #if logMmin > 15 or logM0 > 15 or logM1 > 15: 
        #    return 1e12
        _, p0k, _, _ = Pk_highOm(logMmin, sigma_logM, logM0, logM1, alpha)
        chi2 = np.sum((p0k[8:50] - p0k_obs[8:50])**2)
        print(tt, chi2)
        return chi2

    bestfit_highOm = optimize._minimize_bfgs(
            chi2_P0k_highOm,
            np.array([14.0, 0.38, 14., 14.08, 0.2]),
            maxiter=20,
            eps=0.01)
    print(bestfit_highOm) 
    np.save(os.path.join(dat_dir, 'bestfit_hod_highOm.npy'), bestfit_highOm) 
    return None 


def low_Om(): 
    ''' get best-fit HOD for low Om
    '''
    # pick random low Om cosmology
    i_low_Om = np.random.choice(np.arange(theta_cosmos.shape[0])[theta_cosmos[:,0] < 0.495])

    # import halos 
    halos_low_Om = halos = Halos.Quijote_LHC_HR_Rockstar(i_low_Om, z=0.5)

    def Pk_lowOm(logMmin, sigma_logM, logM0, logM1, alpha):
        hod = {}
        hod['logMmin'] = logMmin
        hod['sigma_logM'] = sigma_logM
        hod['logM0'] = logM0
        hod['logM1'] = logM1
        hod['alpha'] = alpha

        gal = Galaxies.hodGalaxies(halos_low_Om, hod, seed=0)
        _k, _p0k, _p2k, _p4k = CosmoObs.Plk_box(gal)
        return _k[_k < 0.5], _p0k[_k < 0.5], _p2k[_k < 0.5], _p4k[_k < 0.5]


    def chi2_P0k_lowOm(tt):
        logMmin, sigma_logM, logM0, logM1, alpha = tt

        if sigma_logM < 1e-4:
            return 1e12
        if logMmin < 12 or logM0 < 12 or logM1 < 12: 
            return 1e12 
        if logMmin > 15 or logM0 > 15 or logM1 > 15: 
            return 1e12
        _, p0k, _, _ = Pk_lowOm(logMmin, sigma_logM, logM0, logM1, alpha)
        chi2 = np.sum((p0k[8:50] - p0k_obs[8:50])**2)
        print(tt, chi2)
        return chi2

    bestfit_lowOm = optimize._minimize_bfgs(
            chi2_P0k_lowOm,
            np.array([12.5, 0.38, 14., 14.08, 0.5]),
            maxiter=20,
            eps=0.01)
    print(bestfit_lowOm) 
    np.save(os.path.join(dat_dir, 'bestfit_hod_lowOm.npy'), bestfit_lowOm) 
    return None 


if run == 'high_Om': 
    high_Om() 
elif run == 'low_Om': 
    low_Om() 
