# `bin/cmass/sbi/`
scripts for conducting simulation-based inference using the forward modeled training from `bin/cmass/fm/` for the SimBIG CMASS analysis.


- `npe.py`: script for training a neural posterior estimator 

- `npe_opt.py`: script for training neural posterior estimators using the
  optuna hyperparameter optimization framework 

- `valid.py`: script for sampling the ensemble NPEs to run coverage tests for
  validation. 
