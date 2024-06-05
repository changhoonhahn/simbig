import os, sys
import numpy as np

from simbig import halos as Halos
import nbodykit.lab as NBlab

lhc_or_fid  = sys.argv[1]
ireal       = int(sys.argv[2])

print('%s %i' % (lhc_or_fid, ireal)) 

if lhc_or_fid == 'lhc': # latin hypercube 
    dir_halos = '/projects/QUIJOTE/Halos/rockstar/latin_hypercube_HR/'
elif lhc_or_fid == 'fid': # fiducial 
    dir_halos = '/projects/QUIJOTE/Halos/rockstar/fiducial_HR/'
else: 
    raise ValueError

z = 0.5

# rockstar file columns: ID DescID Mvir Vmax Vrms Rvir
# Rs Np X Y Z VX VY VZ JX JY JZ Spin rs_klypin Mvir_all
# M200b M200c M500c M2500c Xoff Voff spin_bullock
# b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c)
# c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) T/|U|
# M_pe_Behroozi M_pe_Diemer Halfmass_Radius
# read in columns: Mvir, Vmax, Vrms, Rvir, Rs, Np,
# X, Y, Z, VX, VY, VZ, parent id
_rstar = np.loadtxt(os.path.join(dir_halos, str(ireal), 'out_0_pid.list'), usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1])

# select only halos 
is_halo = (_rstar[:,-1] == -1)
print('%i of %i are halos' % (np.sum(is_halo), _rstar.shape[0]))
rstar = _rstar[is_halo] 

# calculate concentration Rvir/Rs
conc = rstar[:,3] / rstar[:,4]

# cosmology (Villaesuca-Navarro+2020)
if lhc_or_fid == 'lhc': 
    Om, Ob, h, ns, s8 = Halos.Quijote_LHC_cosmo(ireal)
elif lhc_or_fid == 'fid': 
    Om, Ob, h, ns, s8 = Halos.Quijote_fiducial_cosmo()

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
group_data['Length']    = rstar[:,5].astype(int)
group_data['Position']  = rstar[:,6:9]
group_data['Velocity']  = rstar[:,9:12]  # km/s * (1 + z)
group_data['Mass']      = rstar[:,0]

# calculate velocity offset
group_data['VelocityOffset'] = group_data['Velocity'] * rsd_factor

# save to ArryCatalog for consistency
cat = NBlab.ArrayCatalog(group_data, BoxSize=np.array([1000., 1000., 1000.]))
cat = NBlab.HaloCatalog(cat, cosmo=cosmo, redshift=z, mdef='vir')
cat['Length'] = group_data['Length']
cat['Concentration'] = conc # default concentration is using Dutton & Maccio (2014), which is based only on halo mass.

cat.attrs['Om'] = Om
cat.attrs['Ob'] = Ob
cat.attrs['Ol'] = Ol
cat.attrs['h'] = h
cat.attrs['ns'] = ns
cat.attrs['s8'] = s8
cat.attrs['Hz'] = Hz # km/s/(Mpc/h)A
cat.attrs['rsd_factor'] = rsd_factor

if lhc_or_fid == 'lhc': # latin hypercube 
    cat.save(os.path.join(dir_halos, str(ireal), 'quijote_LHC_HR%i.3.rockstar.bf' % ireal))
elif lhc_or_fid == 'fid': # fiducial 
    cat.save(os.path.join(dir_halos, str(ireal), 'quijote_fid_hr%i.3.rockstar.bf' % ireal))

#if os.path.isdir(os.path.join(dir_halos, str(ireal), 'quijote_fid_hr%i.3.rockstar.bf' % ireal)):
#    # remove all the rockstar files
#    os.system('rm %s%i/halos_0.*' % (dir_halos, ireal))
#    os.system('rm %s%i/out_0.list' % (dir_halos, ireal))
