'''

script for calculating Ngal 


'''
import os, sys 
import numpy as np 

from simbig import util as UT 
from simbig import obs as CosmoObs

np.random.seed(1)


def train_mocks(version='v3', overwrite=False): 
    ''' calculate Ngal of the training dataset 
    '''
    dat_dir = '/tigress/chhahn/simbig/cmass/'
    
    ngals = [] 
    for i_lhc in range(2518): 
        print('LHC %i' % i_lhc)
        for i_hod in range(10):
            fgal = os.path.join(dat_dir, 'train', 
                    'hod.quijote_LH%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_lhc, version, i_hod))
            # read galaxy mock
            gals = UT.rh5_hodcatalog(fgal)

            ngal = len(np.array(gals['Z']))
            ngals.append(ngal)
    np.savetxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.ngal.dat' % version, np.array(ngals)) 
    return None


def test_mocks(version='v3', overwrite=False): 
    ''' calculate Ngal of test data 
    '''
    dat_dir = '/tigress/chhahn/simbig/cmass/'
    
    ngals = [] 
    for i_fid in range(100): 
        print('Fiducial %i' % i_fid)
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.%s.%i.hdf5' % (i_fid, version, i_hod))
            # read galaxy mock
            gals = UT.rh5_hodcatalog(fgal)

            ngal = len(np.array(gals['Z']))
            ngals.append(ngal) 
    
    np.savetxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test.ngal.dat' % version, np.array(ngals)) 
    return None 


def test_fof_mocks(version='v3', overwrite=False): 
    ''' construct test mocks with FOF halo catalogs and Z07 HOD catalogs
    '''
    dat_dir = '/tigress/chhahn/simbig/cmass/'
    
    ngals = [] 
    for i_fid in range(100): 
        print('Fiducial %i' % i_fid)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests_fof',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.%s.fof_z07.%i.hdf5' % (i_fid, version, i_hod))
            
            # read galaxy mock
            gals = UT.rh5_hodcatalog(fgal)

            ngal = len(np.array(gals['Z']))
            ngals.append(ngal)  

    np.savetxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test_fof.ngal.dat' % version, np.array(ngals).T) 
    return None 


def test_abacus_mocks(version='v3', overwrite=False): 
    ''' construct test mocks with abacus halo catalogs and Z07 HOD catalogs
    '''
    ngals = [] 
    for i_fid in range(200): 
        print('Fiducial %i' % i_fid)
        for i_hod in range(5):
            fgal = os.path.join('/tigress/chhahn/simbig/cmass/tests_abacus/',
                    'hod.abacus.c000.z0.500.%i.%s.z07.%i.hdf5' % (i_fid, version, i_hod))
            
            # read galaxy mock
            gals = UT.rh5_hodcatalog(fgal)

            ngal = len(np.array(gals['Z']))
            ngals.append(ngal)  
    np.savetxt('/tigress/chhahn/simbig/simbig.cmass_sgc.%s.test_abacus.ngal.dat' % version, np.array(ngals).T) 
    return None 

train_or_test = str(sys.argv[1])
version = str(sys.argv[2])

if train_or_test == 'train':  
    train_mocks(version=version)
elif train_or_test == 'test': 
    test_mocks(version=version)
    test_fof_mocks(version=version)
    test_abacus_mocks(version=version)
