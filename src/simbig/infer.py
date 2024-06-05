'''

module for different simulation-based inference methods including: 
    - Machine Learning
    - Density Estimation LFI 

'''
import os
import numpy as np 
from tqdm.auto import trange



def training_data(sumstat, kmax, machine='della', nuis=False, version='v3',
        extra=False, restrict=None): 
    ''' load training data 
    '''
    if machine == 'adroit': 
        dat_dir = '/scratch/network/chhahn/simbig/'
    elif machine == 'della': 
        dat_dir = '/tigress/chhahn/simbig/'
    
    str_nuis = ''
    if nuis: str_nuis = '.nuis'

    y_train = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.theta.dat' % (version, str_nuis)), skiprows=1)

    if sumstat in ['p0k', 'p2k', 'p4k', 'p02k', 'plk', 'ng_p0k', 'ng_p02k', 'ng_plk', 'ng_plk_q0k', 'ng_wplk_q0k']: 
        k_plk   = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.k.dat' % version), skiprows=1)
        klim = (k_plk < kmax)
    if 'b0k' in sumstat or 'q0k' in sumstat: 
        k1, k2, k3   = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.k1k2k3.dat' % version), skiprows=1, unpack=True, usecols=[0, 1, 2]) 
        bklim = (k1 < kmax) & (k2 < kmax) & (k3 < kmax) 

    if 'q0k.kmin' in sumstat: 
        kmin = 0.011 # hardcoded kmin to avoid the first bin 
        bklim = bklim & ((k1 > kmin) & (k2 > kmin) & (k3 > kmin))

    if 'ng' in sumstat: 
        ng  = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.ngal.dat' % version))[:,None]

    if sumstat == 'ng': 
        x_train = ng 
    elif sumstat == 'p0k': 
        x_train = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
    elif sumstat == 'p2k': 
        x_train = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
    elif sumstat == 'p4k': 
        x_train = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p4k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
    elif sumstat == 'p02k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        x_train = np.concatenate([p0k, p2k], axis=1)
    elif sumstat == 'plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p4k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        x_train = np.concatenate([p0k, p2k, p4k], axis=1)
    elif sumstat == 'ng_p0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        x_train = np.concatenate([ng, p0k], axis=1)
    elif sumstat == 'ng_p02k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        x_train = np.concatenate([ng, p0k, p2k], axis=1)
    elif sumstat == 'ng_plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p4k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        x_train = np.concatenate([ng, p0k, p2k, p4k], axis=1)
    elif sumstat == 'b0k': 
        x_train = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.b0k.npy' % (version, str_nuis)))[:,bklim]
    elif sumstat == 'ng_b0k': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.b0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'q0k': 
        x_train = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.q0k.npy' % (version, str_nuis)))[:,bklim]
    elif sumstat == 'ng_q0k': 
        q0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.q0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, q0k], axis=1)
    elif sumstat == 'ng_plk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p4k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        q0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.q0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, p0k, p2k, p4k, q0k], axis=1)
    elif sumstat == 'ng_wplk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p0k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p2k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.p4k.dat' % (version, str_nuis)), 
                skiprows=1)[:,klim]
        plk = np.concatenate([p0k, p2k, p4k], axis=1) 

        mu_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.mu_plk.dat' % version))[np.concatenate([klim, klim, klim])]
        sig_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.std_plk.dat' % version))[np.concatenate([klim, klim, klim])]
        plk = (plk - mu_plk)/sig_plk

        q0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.q0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, plk, q0k], axis=1)
    elif sumstat == 'wb0k': 
        x_train = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wb0k.npy' % (version, str_nuis)))[:,bklim]
    elif sumstat == 'ng_wb0k': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wb0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'wq0k': 
        x_train = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wq0k.npy' % (version, str_nuis)))[:,bklim]
    elif sumstat == 'ng_wq0k': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wq0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'wq0k.kmin': 
        x_train = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wq0k.npy' % (version, str_nuis)))[:,bklim]
    elif sumstat == 'ng_wq0k.kmin': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.wq0k.npy' % (version, str_nuis)))[:,bklim]
        x_train = np.concatenate([ng, b0k], axis=1)
    else: 
        x_train = np.loadtxt(
                os.path.join(dat_dir, 'simbig.cmass_sgc.%s%s.%s.kmax%.1f.dat' % (version, str_nuis, sumstat, kmax)), 
                skiprows=1)

    if not extra: 
        y_train = y_train[:20000]
        x_train = x_train[:20000]

    if restrict is not None: 
        if restrict == 'high_logMmin': 
            lim = (y_train[:,5] > 12.5) 
            y_train = y_train[lim]
            x_train = x_train[lim]
        else: 
            raise NotImplementedError
    return y_train, x_train 


def testing_data(sumstat, kmax, test_set, machine='della', version='v3'): 
    ''' testing data 
    '''
    if machine == 'adroit': 
        dat_dir = '/scratch/network/chhahn/simbig/'
    elif machine == 'della': 
        dat_dir = '/tigress/chhahn/simbig/'

    assert test_set in ['test', 'test_fof', 'test_abacus']

    # test parameters --- i.e. true parameters
    y_test = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.theta.dat' % (version, test_set)), skiprows=1)
    # load test data
    if sumstat in ['p0k', 'p2k', 'p4k', 'p02k', 'plk', 'ng_p0k', 'ng_p02k', 'ng_plk', 'ng_plk_q0k', 'ng_wplk_q0k']: 
        k_plk   = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.k.dat' % version), skiprows=1)
        klim = (k_plk < kmax)
    if 'b0k' in sumstat or 'q0k' in sumstat: 
        k1, k2, k3   = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.k1k2k3.dat' % version), skiprows=1, unpack=True, usecols=[0, 1, 2]) 
        bklim = (k1 < kmax) & (k2 < kmax) & (k3 < kmax)
    if 'q0k.kmin' in sumstat: 
        kmin = 0.011 # hardcoded kmin to avoid the first bin 
        bklim = bklim & ((k1 > kmin) & (k2 > kmin) & (k3 > kmin))

    if 'ng' in sumstat: 
        ng  = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.ngal.dat' % (version, test_set)))[:,None]

    if sumstat == 'plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p4k.dat' % (version, test_set)), skiprows=1)[:,klim]
        x_test = np.concatenate([p0k, p2k, p4k], axis=1)
    elif sumstat == 'p02k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        x_test = np.concatenate([p0k, p2k], axis=1)
    elif sumstat == 'p0k': 
        x_test  = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
    elif sumstat == 'p2k': 
        x_test  = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
    elif sumstat == 'p4k': 
        x_test  = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p4k.dat' % (version, test_set)), skiprows=1)[:,klim]
    elif sumstat == 'ng': 
        x_test  = ng
    elif sumstat == 'ng_plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p4k.dat' % (version, test_set)), skiprows=1)[:,klim]
        x_test = np.concatenate([ng, p0k, p2k, p4k], axis=1)
    elif sumstat == 'ng_p02k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        x_test = np.concatenate([ng, p0k, p2k], axis=1)
    elif sumstat == 'ng_p0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        x_test = np.concatenate([ng, p0k], axis=1)
    elif sumstat == 'b0k': 
        x_test = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.b0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
    elif sumstat == 'ng_b0k': 
        b0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.b0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
        x_test = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'q0k': 
        x_test = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.q0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
    elif sumstat == 'ng_q0k': 
        q0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.q0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
        x_test = np.concatenate([ng, q0k], axis=1)
    elif sumstat == 'ng_plk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p4k.dat' % (version, test_set)), skiprows=1)[:,klim]
        q0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.q0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
        x_test = np.concatenate([ng, p0k, p2k, p4k, q0k], axis=1)
    elif sumstat == 'ng_wplk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p0k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p2k.dat' % (version, test_set)), skiprows=1)[:,klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.p4k.dat' % (version, test_set)), skiprows=1)[:,klim]
        plk = np.concatenate([p0k, p2k, p4k], axis=1) 

        mu_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.mu_plk.dat' % version))[np.concatenate([klim, klim, klim])]
        sig_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.std_plk.dat' % version))[np.concatenate([klim, klim, klim])]
        plk = (plk - mu_plk)/sig_plk

        q0k = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.q0k.dat' % (version, test_set)), skiprows=1)[:,bklim]
        x_test = np.concatenate([ng, plk, q0k], axis=1)
    elif sumstat == 'wb0k': 
        x_test = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wb0k.npy' % (version, test_set)))[:,bklim]
    elif sumstat == 'ng_wb0k': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wb0k.npy' % (version, test_set)))[:,bklim]
        x_test = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'wq0k': 
        x_test = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wq0k.npy' % (version, test_set)))[:,bklim]
    elif sumstat == 'ng_wq0k': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wq0k.npy' % (version, test_set)))[:,bklim]
        x_test = np.concatenate([ng, b0k], axis=1)
    elif sumstat == 'wq0k.kmin': 
        x_test = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wq0k.npy' % (version, test_set)))[:,bklim]
    elif sumstat == 'ng_wq0k.kmin': 
        b0k = np.load(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.wq0k.npy' % (version, test_set)))[:,bklim]
        x_test = np.concatenate([ng, b0k], axis=1)
    else: 
        ftest   = os.path.join(dat_dir, 'simbig.cmass_sgc.%s.%s.%s.kmax%.1f.dat' % (version, test_set, sumstat, kmax))
        x_test  = np.loadtxt(ftest, skiprows=1)

    return y_test, x_test 


def obs_data(sumstat, kmax, machine='della', version='v4'): 
    ''' observational data  
    '''
    if machine == 'adroit': 
        dat_dir = '/scratch/network/chhahn/simbig/'
    elif machine == 'della': 
        dat_dir = '/tigress/chhahn/simbig/'

    # load test data
    if sumstat in ['p0k', 'p2k', 'p4k', 'p02k', 'plk', 'ng_p0k', 'ng_p02k', 'ng_plk', 'ng_plk_q0k']: 
        k_plk   = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.k.dat'), skiprows=1)
        klim = (k_plk < kmax)
    if 'b0k' in sumstat or 'q0k' in sumstat: 
        k1, k2, k3   = np.loadtxt(os.path.join(dat_dir,
            'obs.cmass_sgc.k1k2k3.w_nofc.dat'), skiprows=1, unpack=True, usecols=[0, 1, 2]) 
        bklim = (k1 < kmax) & (k2 < kmax) & (k3 < kmax)

    if 'ng' in sumstat: 
        ng  = [np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.ngal.w_nofc.dat'))] 

    if sumstat == 'plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p0k.w_nofc.dat'), skiprows=1)[klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p2k.w_nofc.dat'), skiprows=1)[klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p4k.w_nofc.dat'), skiprows=1)[klim]
        x_obs = np.concatenate([p0k, p2k, p4k])
    elif sumstat == 'ng_plk': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p0k.w_nofc.dat'), skiprows=1)[klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p2k.w_nofc.dat'), skiprows=1)[klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p4k.w_nofc.dat'), skiprows=1)[klim]
        x_obs = np.concatenate([ng, p0k, p2k, p4k])
    elif sumstat == 'q0k': 
        x_obs = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.q0k.w_nofc.dat'), skiprows=1)[bklim]
    elif sumstat == 'ng_q0k': 
        q0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.q0k.w_nofc.dat'), skiprows=1)[bklim]
        x_obs = np.concatenate([ng, q0k])
    elif sumstat == 'ng_plk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p0k.w_nofc.dat'), skiprows=1)[klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p2k.w_nofc.dat'), skiprows=1)[klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p4k.w_nofc.dat'), skiprows=1)[klim]
        q0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.q0k.w_nofc.dat'), skiprows=1)[bklim]
        x_obs = np.concatenate([ng, p0k, p2k, p4k, q0k])
    elif sumstat == 'ng_plk_q0k': 
        p0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p0k.w_nofc.dat'), skiprows=1)[klim]
        p2k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p2k.w_nofc.dat'), skiprows=1)[klim]
        p4k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.p4k.w_nofc.dat'), skiprows=1)[klim]
        plk = np.concatenate([p0k, p2k, p4k]) 
        mu_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.mu_plk.dat' % version))[:,klim]
        std_plk = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.%s.std_plk.dat' % version))[:,klim]
        plk = (plk - mu_plk)/sig_plk

        q0k = np.loadtxt(os.path.join(dat_dir, 'obs.cmass_sgc.q0k.w_nofc.dat'), skiprows=1)[bklim]

        x_obs = np.concatenate([ng, plk, q0k])
    return x_obs 


def npe_validation(qphi, x_test, y_test, device='cpu', n_sample=10000): 
    ''' validation of NPE
    '''
    import torch 

    mcmcs, pp_thetas, rank_thetas = [], [], []
    t = trange(x_test.shape[0], leave=False)
    for i in t: 
        _x_test = x_test[i]
        _y_test = y_test[i]

        # sample posterior p(theta | x_test_i)
        mcmc_npe = qphi.sample((n_sample,), 
                x=torch.as_tensor(_x_test.astype(np.float32)).to(device), 
                show_progress_bars=False)
        mcmc_npe = np.array(mcmc_npe.detach().to('cpu'))

        # apply importance weights
        #w_mcmc = np.ones(10000)
        #w_mcmc[mcmc_npe[:,0] > 0.4] = 2018./1500.  # upweight samples with Om > 0.4 
        #w_mcmc *= 10000./np.sum(w_mcmc)

        # calculate percentile score and rank
        rank_theta = []
        for itheta in range(5): 
            rank_theta.append(np.sum(mcmc_npe[:,itheta] < _y_test[itheta]))
            #rank_theta.append(np.sum(w_mcmc[mcmc_npe[:,itheta] < _y_test[itheta]]))

        mcmcs.append(mcmc_npe) 
        rank_thetas.append(rank_theta)

        t.set_description('%i' % i, refresh=False)
    
    return np.array(mcmcs), np.array(rank_thetas)
