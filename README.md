# SIMBIG
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/changhoonhahn/molino/blob/main/LICENSE)
[![Gitter](https://badges.gitter.im/boss_sbi/community.svg)](https://gitter.im/boss_sbi/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Simulation-based Inference of Galaxies


## Installation 
First [set up a new anaconda environment](#setting-up-a-conda-environment) to
avoid any package conflicts. 

Activate the conda environment and then install the `simbig` package
```bash
# first clone the repo
git clone https://github.com/changhoonhahn/simbig.git

# go to the repo
cd simbig 

# install the package 
pip install -e . 
```

You will also need to set the `$QUIJOTE_DIR` environment variable. To do this, add the following line 
```
# if you're on tiger
export QUIJOTE_DIR="/projects/QUIJOTE/"

# if you're on flatiron popeye  
export QUIJOTE_DIR="/mnt/home/chahn/ceph/QUIJOTE/"
```
to your `~/.bashrc` file. If you don't know how to do this then just copy paste the following: 
```
# if you're on tiger
echo 'export QUIJOTE_DIR="/projects/QUIJOTE/"' >> ~/.bashrc

# if you're on flatiron popeye  
echo 'export QUIJOTE_DIR="/mnt/home/chahn/ceph/QUIJOTE/"' >> ~/.bashrc
```

Once you've added the line, don't forget to run
```
source ~/.bashrc
```

### Setting Up a Conda Environment 
#### On `tiger` 
If you're on Princeton's `tiger` cluster, you don't have to install anaconda.
You can load it using 
```bash
module load anaconda 
```

Afterwards you can create a new conda environment using
```bash
conda create -n ENV_NAME_HERE python=3.7 ipython 
```
and following the instructions. 


To activate the conda environment you created
```
conda activate ENV_NAME_HERE 
```

Later, if you want to exist the conda environemtn
```bash
conda deactivate 
```

### `pySpectrum` installation
If you're interested in calculating the bispectrum, you'll have to install [`pySpectrum`](https://github.com/changhoonhahn/pySpectrum). 
More specifically, you'll have to install the `survey` branch of `pySpectrum`, which includes the latest
updates (this will be merged into main branch soon).

To install `pySpectrum`, activate your conda environment and run: 
```bash
git clone --single-branch --branch survey https://github.com/changhoonhahn/pySpectrum.git
cd pySpectrum
pip install -e . 
```

### Dependencies
The `simbig` package requires the following python pacakges: 
- [nbodykit](https://nbodykit.readthedocs.io/) 
- [pymangle](https://github.com/esheldon/pymangle)
- [sbi](https://github.com/mackelab/sbi/)

**tl;dr** Run the following lines after activating the conda environment 
```
conda install -c bccp nbodykit
pip install pymangle
pip install sbi 
```


## Generating an HOD catalog for HR Quijote LHC 
```python
import numpy as np 
from simbig.halos import Quijote_LHC_HR
from simbig import galaxies as Galaxies

# read in halo catalog 
halos = Quijote_LHC_HR(1, z=0.5)

# get LOWZ HOD parameters
theta_hod = Galaxies.thetahod_lowz_ngc()

# apply HOD 
gals = Galaxies.hodGalaxies(halos, theta_hod, seed=0) 
print(np.array(gals['Position']))
```

## Resources 
Details and resources on the original BOSS analysis: [https://sites.google.com/view/learningouruniverse/boss-analysis?authuser=0](https://sites.google.com/view/learningouruniverse/boss-analysis?authuser=0)


