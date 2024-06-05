'''

script to submit jobs to della 


'''
import os, sys 
import numpy as np 


def postprocess_quij_lhc_hr(i0, i1): 
    ''' postprocess LHC rockstar output 
    '''
    dir_halos = '/projects/QUIJOTE/Halos/rockstar/latin_hypercube_HR/'
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J post.q.lhc.%i_%i' % (i0, i1), 
        '#SBATCH --time=06:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/post.q.lhc.%i_%i' % (i0, i1),
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '',
        ''])
        
    n = 0
    for i in range(i0, i1+1): 
        if os.path.isdir(os.path.join(dir_halos, str(i), 'quijote_LHC_HR%i.3.rockstar.bf' % i)):
            continue 
        a += 'python /home/chhahn/projects/simbig/cmass/bin/postprocess_rockstar.py lhc %i\n' % i
        n += 1 
    if n == 0: 
        return None 
    
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    #os.system('rm script.slurm')
    return None 


def postprocess_quij_fid_hr(i0, i1): 
    ''' postprocess fid rockstar output 
    '''
    dir_halos = '/projects/QUIJOTE/Halos/rockstar/fiducial_HR/'
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J post.q.fid.%i_%i' % (i0, i1), 
        '#SBATCH --time=06:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/post.q.fid.%i_%i' % (i0, i1),
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '',
        ''])
        
    n = 0
    for i in range(i0, i1): 
        if os.path.isdir(os.path.join(dir_halos, str(i), 'quijote_fid_hr%i.3.rockstar.bf' % i)):
            continue 
        a += 'python /home/chhahn/projects/simbig/cmass/bin/postprocess_rockstar.py fid %i\n' % i
        n += 1 
    if n == 0: 
        return None 
    
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def postprocess_abacus(i0, i1): 
    ''' postprocess fid rockstar output 
    '''
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J post.abacus.%i_%i' % (i0, i1), 
        '#SBATCH --time=06:00:00',
        "#SBATCH --export=ALL",
        '#SBATCH -o o/post.abacus.%i_%i' % (i0, i1),
        '',
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        '',
        ''])
        
    for i in range(i0, i1): 
        a += 'python /home/chhahn/projects/simbig/cmass/bin/postprocess_abacus.py %i\n' % i
    
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(a)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def theta_train(version='v2'): 
    ''' submit jobs to construct HOD parameters 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J theta_%s" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=00:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_theta_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py theta_train %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def cmass_train(i0, i1, version='v2', overwrite=False): 
    ''' submit jobs to construct HOD CMASS training catalog 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cmass_%s_%i_%i" % (version, i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --mem=32G", 
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_cmass_train_%s_%i_%i.o" % (version, i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py train %i %i %s %s" % (i0, i1, version, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def theta_test(version='v3'): 
    ''' submit jobs to construct HOD parameters 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J theta_%s" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=00:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_theta_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py theta_test %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def cmass_test(i0, i1, version='v3', overwrite=False): 
    ''' submit jobs to construct test SIMBIG CMASS catalogs at the fiducial cosmology
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cmass_test_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --time=12:00:00",
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/cmass_test_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python make_simbig.py test %i %i %s %s" % (i0, i1, version, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def theta_test_fof(version='v3'): 
    ''' submit jobs to construct HOD parameters 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J theta_fof_%s" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=00:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_theta_fof_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py theta_test_fof %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def cmass_test_fof(i0, i1, version='v3', overwrite=False): 
    ''' submit jobs to construct test SIMBIG CMASS catalogs at the fiducial cosmology
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cmass_test_fof_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --time=12:00:00",
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/cmass_test_fof_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python make_simbig.py test_fof %i %i %s %s" % (i0, i1, version, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def theta_test_abacus(version='v3'): 
    ''' submit jobs to construct HOD parameters 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J theta_abacus_%s" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=00:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_theta_abacus_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py theta_test_abacus %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def cmass_test_abacus(i0, i1, version='v3', overwrite=False): 
    ''' submit jobs to construct test SIMBIG CMASS catalogs at the fiducial cosmology
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cmass_test_abacus_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --time=12:00:00",
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/cmass_test_abacus_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python make_simbig.py test_abacus %i %i %s %s" % (i0, i1, version, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def cmass_test2(i0, i1, overwrite=False): 
    ''' submit jobs to construct test SIMBIG CMASS catalogs at the fiducial cosmology
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J cmass_test_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --time=12:00:00",
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/cmass_test_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python make_simbig_v1.py test2 %i %i %s" % (i0, i1, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


def halo_train(i0, i1): 
    ''' submit jobs to construct v1 HOD CMASS training catalog 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J halo_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --mem=16G", 
        "#SBATCH --time=12:00:00",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_halo_train_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig_v1.py train_halo %i %i" % (i0, i1), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def halo_test(i0, i1): 
    ''' submit jobs to construct v1 HOD CMASS training catalog 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J halo_%i_%i" % (i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --mem=16G", 
        "#SBATCH --time=12:00:00",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_halo_test_%i_%i.o" % (i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig_v1.py test_halo %i %i" % (i0, i1), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def hod_bestfit(cosmo): 
    ''' submit jobs to construct v1 HOD CMASS training catalog 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J hod_%s" % cosmo,
        "#SBATCH --nodes=1",
        "#SBATCH --mem=16G", 
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_hod_%s.o" % cosmo,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python hod_priors.py %s" % cosmo, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def train_nbar(version='v4'): 
    ''' submit jobs to measure B0(k) of the training data 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_ngal" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_ngal_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python ngal.py train %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def test_nbar(version='v4'): 
    ''' submit jobs to measure B0(k) of the training data 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s_ngal" % version,
        "#SBATCH --nodes=1",
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_ngal_%s.o" % version,
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python ngal.py test %s" % version, 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None



def train_b0k(i0, i1, version='v2', overwrite=False): 
    ''' submit jobs to measure B0(k) of the training data 
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J b0k_%s_%i_%i" % (version, i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --mem=14G", 
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_b0k_train_%s_%i_%i.o" % (version, i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py train_b0k %i %i %s %s" % (i0, i1, version, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def test_b0k(i0, i1, test_set, version='v4', overwrite=False): 
    ''' submit jobs to measure B0(k) of the test data sets
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J b0k_%s_%s_%i_%i" % (test_set, version, i0, i1),
        "#SBATCH --nodes=1",
        "#SBATCH --mem=14G", 
        "#SBATCH --time=11:59:59",
        "#SBATCH --export=ALL",
        "#SBATCH --output=o/_b0k_%s_%s_%i_%i.o" % (test_set, version, i0, i1),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        "module load anaconda3/2021.11", 
        "conda activate sbi", 
        "",
        "python make_simbig.py test_b0k %i %i %s %s %s" % (i0, i1, version, test_set, ['False', 'True'][overwrite]), 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None




######################################################
# post process halo catalogs
######################################################
#for i in range(20): 
#    postprocess_quij_fid_hr(i*5, (i+1)*5)

#for i in range(20): 
#    postprocess_abacus(i*10, (i+1)*10)
 
######################################################
# determine prior range 
######################################################
#hod_bestfit('high_Om') 
#hod_bestfit('low_Om') 

######################################################
# construct training data 
######################################################
#theta_train(version='v4') # generate HOD parameters
#for i in range(100):
#    cmass_train(20*i, 20*(i+1)-1, version='v4')
#for i in range(100, 126):
#    cmass_train(20*i, 20*(i+1)-1, version='v4')
#cmass_train(2500, 2518, version='v4') 
#cmass_train(2060, 2079, version='v4') 
######################################################
# construct test data 
######################################################
#theta_test(version='v4') # generate HOD parameters
#cmass_test(1, 3, version='v4') 
#for i in range(1, 25): 
#    cmass_test(i*4, (i+1)*4-1, version='v4') 

# FOF data 
#theta_test_fof(version='v4') 
#for i in range(25): 
#    cmass_test_fof(i*4, (i+1)*4-1, version='v4') 

# Abacus data
#theta_test_abacus(version='v4')
#for i in range(25): 
#    cmass_test_abacus(i*8, (i+1)*8, version='v4')
######################################################
# measure nbar(k) 
######################################################
train_nbar(version='v4') 

######################################################
# measure B0(k) 
######################################################
#train_b0k(1, 9, version='v4', overwrite=False)
#for i in range(1,252): 
#    train_b0k(i*10, (i+1)*10-1, version='v4', overwrite=False)

#test_b0k(1, 3, 'TEST0', version='v4', overwrite=False) 
#for i in range(1, 25): 
#    test_b0k(4*i, 4*(i+1)-1, 'TEST0', version='v4', overwrite=False) 
#
#test_b0k(1, 3, 'TEST1', version='v4', overwrite=False) 
#for i in range(1, 25): 
#    test_b0k(4*i, 4*(i+1)-1, 'TEST1', version='v4', overwrite=False) 
#
#test_b0k(1, 3, 'TEST2', version='v4', overwrite=False) 
#for i in range(1, 50): 
#    test_b0k(4*i, 4*(i+1)-1, 'TEST2', version='v4', overwrite=False) 

######################################################
# archive
######################################################
#for i in range(25): 
#    cmass_test2(i*4, (i+1)*4-1, overwrite=True) 

#for i in range(1, 10):
#    halo_train(i*200, (i+1)*200-1)
#halo_test(0, 19)
#for i in range(1, 5):
#    halo_test(i*20, (i+1)*20-1)
