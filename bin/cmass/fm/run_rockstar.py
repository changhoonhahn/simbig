'''

script for deploying rockstar on Quijtoe HR latin hyper cube snapshots 
to construt halo catalogs 


'''
import os, sys
import numpy as np


dir_rockstar = '/mnt/home/chahn/projects/rockstar/'
dir_snapshots = '/mnt/ceph/users/fvillaescusa/Quijote/Snapshots/latin_hypercube_HR/'
dir_halos = '/mnt/home/chahn/ceph/Halos/latin_hypercube_HR/'

def quijote_HR_LHC(i0, i1):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J quijote',
        '#SBATCH --ntasks-per-node=28',
        '#SBATCH --time=23:59:59',
        '#SBATCH --partition=cca', 
        '#SBATCH -o o/rockstar.%i_%i' % (i0, i1),
        '',# 'module load intel-mpi/2017.4.196', 'module load openmpi',
        'module load gcc lib/hdf5',
        '',  
        'source ~/.bashrc',
        'conda activate simbig',
        ''])
    
    n_run = 0 
    for i_lhc in range(i0, i1): 
        fbf = os.path.join(dir_halos, str(i_lhc), 'quijote_LHC_HR%i.3.rockstar.bf' % i_lhc)

        if not os.path.isdir(os.path.join(dir_halos, str(i_lhc))): 
            os.system('mkdir %s' % os.path.join(dir_halos, str(i_lhc)))

        #if os.path.isdir(fbf) or os.path.isfile(fbf): 
        #    print('  %s exists' % fbf) 
        #    continue 

        if not os.path.isfile(os.path.join(dir_halos, str(i_lhc), "out_0.list")):
            # first write config file for given simulation 
            a = '\n'.join([
                '#Rockstar Halo Finder',
                'FILE_FORMAT = "AREPO"',
                'PARTICLE_MASS = 0       # must specify (in Msun/h) for ART or ASCII',
                '',
                '# You should specify cosmology parameters only for ASCII formats',
                '# For GADGET2 and ART, these parameters will be replaced with values from the',
                '# particle data file',
                '',
                '# For AREPO / GADGET2 HDF5, you would use the following instead:',
                '# Make sure to compile with "make with_hdf5"!',
                'AREPO_LENGTH_CONVERSION = 1e-3',
                'AREPO_MASS_CONVERSION = 1e+10',
                '',
                'MASS_DEFINITION = "vir" ',
                '',
                '#This specifies the use of multiple processors:',
                'PARALLEL_IO=1',
                'PERIODIC = 1',
                '',
                'FORCE_RES = 0.05 #Force resolution of simulation, in Mpc/h',
                '',
                'MIN_HALO_OUTPUT_SIZE = 20 ',
                '',
                'BOX_SIZE = 1000.00 #Mpc',
                '',
                'INBASE = "%s%i"' % (dir_snapshots, i_lhc),
                'FILENAME="snapdir_003/snap_003.<block>.hdf5"',# 'STARTING_SNAP = 0 ', 'NUM_SNAPS = 5 ',
                'NUM_BLOCKS=8',
                '',
                'OUTBASE = "%s%i"' % (dir_halos, i_lhc),
                '',
                'NUM_READERS = 1',
                'NUM_WRITERS = 8',
                'FORK_READERS_FROM_WRITERS = 1',
                'FORK_PROCESSORS_PER_MACHINE = 8'])
            f = open(os.path.join(dir_halos, str(i_lhc), 'quijote_fid_hr.%i.cfg' % i_lhc), 'w')
            f.write(a)
            f.close() 
        
            # next, write slurm file for submitting the job
            slurm += '\n'.join([
                '', 
                'echo "%s"' % i_lhc, 
                'dir_rockstar="%s" # rockstar repo directory' % dir_rockstar,
                'dir_snapshot="%s%i" # output directory' % (dir_halos, i_lhc),
                '',
                'mkdir -p $dir_snapshot',
                '',
                '$dir_rockstar/rockstar -c $dir_snapshot/quijote_fid_hr.%i.cfg &> $dir_snapshot/server%i.dat &' % (i_lhc, i_lhc),
                '',
                'while [ ! -f "$dir_snapshot/auto-rockstar.cfg" ]; do echo "sleeping"; sleep 1; done',
                '',
                '# deploy jobs ',
                'mpirun $dir_rockstar/rockstar -c $dir_snapshot/auto-rockstar.cfg >> $dir_snapshot/output.dat 2>&1',
                '', 
                ''])
            n_run += 1

    if n_run == 0: return None

    f = open('rockstar_quij_hr_lhc.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch rockstar_quij_hr_lhc.%i_%i.slurm' % (i0, i1))
    #os.system('rm rockstar_quijote_hr_fid.%i_%i.slurm' % (i0, i1))
    return None


def quijote_HR_LHC_pid(i0, i1):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J quijote',
        '#SBATCH --ntasks-per-node=28',
        '#SBATCH --time=23:59:59',
        '#SBATCH --partition=cca', 
        '#SBATCH -o o/rockstar.%i_%i' % (i0, i1),
        '',# 'module load intel-mpi/2017.4.196', 'module load openmpi',
        'module load gcc lib/hdf5',
        '',  
        'source ~/.bashrc',
        'conda activate simbig',
        ''])
    
    n_run = 0 
    for i_lhc in range(i0, i1): 
        fbf = os.path.join(dir_halos, str(i_lhc), 'quijote_LHC_HR%i.3.rockstar.bf' % i_lhc)

        if not os.path.isdir(os.path.join(dir_halos, str(i_lhc))): 
            os.system('mkdir %s' % os.path.join(dir_halos, str(i_lhc)))

        # next, write slurm file for submitting the job
        slurm += '\n'.join([
            '', 
            'echo "%s"' % i_lhc, 
            'dir_rockstar="%s" # rockstar repo directory' % dir_rockstar,
            'dir_snapshot="%s%i" # output directory' % (dir_halos, i_lhc),
            '',
            '$dir_rockstar/util/find_parents $dir_snapshot/out_0.list 1000 > $dir_snapshot/out_0_pid.list', 
            #'', #'python /mnt/home/chahn/projects/simbig/cmass/bin/postprocess_rockstar.py %i' % i_lhc, 
            ''])
        n_run += 1

    if n_run == 0: return None

    f = open('rockstar_quij_hr_lhc.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch rockstar_quij_hr_lhc.%i_%i.slurm' % (i0, i1))
    #os.system('rm rockstar_quijote_hr_fid.%i_%i.slurm' % (i0, i1))
    return None


#quijote_HR_LHC(4, 100) 
for i in range(1, 20): 
   quijote_HR_LHC(i*100, (i+1)*100)
