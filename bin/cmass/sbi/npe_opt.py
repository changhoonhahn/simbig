'''

script to train Neural Posterior Estimators with Optuna hyperparameter optimization

'''
import os, sys 
import numpy as np
#from scipy.stats import ks_2samp as KStest

from simbig import nns as NNs 
from simbig import infer as Infer

import torch
from torch import nn 
from torch.utils.tensorboard.writer import SummaryWriter

import optuna 

from sbi import utils as Ut
from sbi import inference as Inference

import matplotlib.pyplot as plt 


output_dir = '/tigress/chhahn/simbig/npe/'


sumstat     = sys.argv[1]
kmax        = float(sys.argv[2]) 
nuis        = (sys.argv[3] == 'True') 
machine     = sys.argv[4]
compress    = (sys.argv[5] == 'True') 
nf_model    = sys.argv[6]
only_cosmo  = (sys.argv[7] == 'True') 

##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

seed = 12387
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
##################################################################################
# load data 
##################################################################################
version = 'v4'
y_train, x_train = Infer.training_data(sumstat, kmax, 
        machine=machine, nuis=nuis, version=version, 
        restrict=None, extra=True) 
print('Ntrain=%i' % x_train.shape[0])

y_valid = y_train[20000:][::10]
x_valid = x_train[20000:][::10]

y_train = y_train[:20000]
x_train = x_train[:20000]

if only_cosmo: 
    y_valid = y_valid[:,:5]
    y_train = y_train[:,:5]
##################################################################################
# set prior 
##################################################################################
if nuis: 
    # om, ob, h, ns, s8, HOD parameters, Ashot 
    lower_bounds = torch.tensor([0.1, 0.03, 0.5, 0.8, 0.6, 12., 0.1, 13., 13., 0.0, -1., 0.2, 0., 0.2, -1e4])
    upper_bounds = torch.tensor([0.5, 0.07, 0.9, 1.2, 1., 14., 0.6, 15., 15., 1.5, 1., 2.0, 0.7, 2.0, 1e4])
else: 
    # om, ob, h, ns, s8, HOD parameters, Ashot 
    lower_bounds = torch.tensor([0.1, 0.03, 0.5, 0.8, 0.6, 12., 0.1, 13., 13., 0.0, -1., 0.2, 0., 0.2])
    upper_bounds = torch.tensor([0.5, 0.07, 0.9, 1.2, 1., 14., 0.6, 15., 15., 1.5, 1., 2.0, 0.7, 2.0])

if only_cosmo: 
    lower_bounds = lower_bounds[:5]
    upper_bounds = upper_bounds[:5]

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

##################################################################################
# OPTUNA
##################################################################################
# Optuna Parameters
if int(kmax * 100) % 10 == 0: 
    kmax_str = 'kmax%.1f' % kmax
else:  
    kmax_str = 'kmax%.2f' % kmax

n_trials    = 1000
study_name  = 'qphi.exp1.%s.%s%s%s.%s%s.%s' % (nf_model, version, 
        ['', '.only_cosmo'][only_cosmo], ['', '.nuis'][nuis], ['', 'c'][compress], sumstat, kmax_str)
n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 10
n_transf_min, n_transf_max = 2, 10
n_hidden_min, n_hidden_max = 64, 512
n_lr_min, n_lr_max = 5e-6, 1e-3 
p_drop_min, p_drop_max = 0., 1.
#clip_max_min, clip_max_max = 1., 5.

# compression 
n_comp_min, n_comp_max = 64, 512 
n_layers_mlp_min, n_layers_mlp_max = 2, 5 
n_hidden_mlp_min, n_hidden_mlp_max = 64, 512 


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 
    p_drop = trial.suggest_float("p_drop", p_drop_min, p_drop_max)
    #clip_max = trial.suggest_float("clip_max_norm", clip_max_min, clip_max_max) 
    #use_batch_norm = True #trial.suggest_categorical('use_batch_norm', [True, False]) 

    if compress: 
        n_comp = trial.suggest_int('ncomp', n_comp_min, n_comp_max)
        n_layers_mlp = trial.suggest_int("n_layers_mlp", n_layers_mlp_min, n_layers_mlp_max)
        n_hidden_mlp = trial.suggest_int("n_hidden_mlp", n_hidden_mlp_min, n_hidden_mlp_max, 
                log=True)
        
        #  MLP for compression 
        mlp = NNs.MLP(x_train.shape[1], n_comp, [n_hidden_mlp for _ in range(n_layers_mlp)])
    else: 
        mlp = nn.Identity()
    
    neural_posterior = Ut.posterior_nn(nf_model, 
            embedding_net=mlp, 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            dropout_probability=p_drop, 
            use_batch_norm=True
            )

    anpe = Inference.SNPE(prior=prior,
            density_estimator=neural_posterior,
            device=device, 
            summary_writer=SummaryWriter('%s/%s/%s.%i' % 
                (output_dir, study_name, study_name, trial.number)))

    anpe.append_simulations(
        torch.as_tensor(y_train.astype(np.float32)).to(device),
        torch.as_tensor(x_train.astype(np.float32)).to(device))

    p_theta_x_est = anpe.train(
            training_batch_size=50,
            learning_rate=lr, #clip_max_norm=clip_max, 
            show_train_summary=True)

    # save trained NPE  
    qphi    = anpe.build_posterior(p_theta_x_est)
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)
        
    best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]

    anpe._summary_writer.add_hparams(
            {'n_blocks': n_blocks, 'n_transf': n_transf, 'n_hidden': n_hidden, 'lr': lr, 'p_drop': p_drop},
            {'best_valid_log_prob': best_valid_log_prob}
            )
        
    # calculate rank statistics 
    # get ranks for validation 
    #_, ranks = Infer.npe_validation(qphi, x_valid, y_valid, device=device, n_sample=1000)

    #fig = plt.figure(figsize=(6,6))
    #sub = fig.add_subplot(111)
    #_ = sub.hist(ranks[:,0]/1000., range=(0, 1), bins=20, density=True, histtype='step', linewidth=2)
    #_ = sub.hist(ranks[:,4]/1000., range=(0, 1), bins=20, density=True, histtype='step', linewidth=2)
    #sub.plot([0., 1.], [1., 1.], c='k', ls='--', linewidth=2)
    #sub.set_xlabel('rank statistics', fontsize=25)
    #sub.set_xlim(0., 1.)
    #sub.set_ylim(0., None)
    #sub.set_yticks([])
    #fig.savefig(fqphi.replace('.pt', '.rank.png'), bbox_inches='tight') 
    #plt.close() 

    #om_ks = KStest(unif, ranks[:,0]/1000.)
    #s8_ks = KStest(unif, ranks[:,4]/1000.) 

    return -1*best_valid_log_prob#, om_ks.statistic, s8_ks.statistic


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) # multivariate=True)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) # , "minimize", "minimize"

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
