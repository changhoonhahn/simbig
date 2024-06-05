'''

script to postprocess abacus halo catalogs


'''
import os, sys
import h5py 
import numpy as np

from simbig import halos as Halos
import nbodykit.lab as NBlab

ireal       = int(sys.argv[1])

i_ph    = (ireal // 8) 
i_box   = (ireal % 8) 
print('ph%i box%i' % (i_ph, i_box)) 

dir_halos = '/tigress/chhahn/simbig/abacus/'

z = 0.5

# read data 
abacus = h5py.File(os.path.join(dir_halos, 'AbacusSummit_base_c000_ph%s.z0.500.%i.lite.hdf5' % (str(i_ph).zfill(3), i_box)), 'r') 

# cosmology (see https://abacussummit.readthedocs.io/en/latest/cosmologies.html)
h   = 0.6736  
Om  = abacus.attrs['Omega_M'] 
Ob  = abacus.attrs['omega_b'] * h**-2
ns  = abacus.attrs['n_s'] 
s8  = 0.807952

# define cosmology; caution: we don't match sigma8 here
cosmo = NBlab.cosmology.Planck15.clone(
        h=h,
        Omega0_b=Ob,
        Omega0_cdm=Om - Ob,
        m_ncdm=None,
        n_s=ns)
Ol = 1.  - Om
Hz = 100.0 * np.sqrt(Om * (1. + z)**3 + Ol) # km/s/(Mpc/h)

rsd_factor = (1. + z) / Hz

group_data = {}
group_data['Length']    = abacus['Length'][...]
group_data['Position']  = abacus['Position'][...]
group_data['Velocity']  = abacus['Velocity'][...] 
group_data['Mass']      = abacus['Mass'][...]

# calculate velocity offset
group_data['VelocityOffset'] = group_data['Velocity'] * rsd_factor

# save to ArryCatalog for consistency
cat = NBlab.ArrayCatalog(group_data, BoxSize=np.array([1000., 1000., 1000.]))
cat = NBlab.HaloCatalog(cat, cosmo=cosmo, redshift=z, mdef='vir')
cat['Length'] = group_data['Length']
cat['Concentration'] = abacus['Concentration'][...]

cat.attrs['Om'] = Om
cat.attrs['Ob'] = Ob
cat.attrs['Ol'] = Ol
cat.attrs['h'] = h
cat.attrs['ns'] = ns
cat.attrs['s8'] = s8
cat.attrs['Hz'] = Hz # km/s/(Mpc/h)A
cat.attrs['rsd_factor'] = rsd_factor

cat.save(os.path.join(dir_halos, 'abacus.c000.z0.500.%i.bf' % ireal))
