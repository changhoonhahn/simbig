'''

script to train Neural Posterior Estimators 

'''
import os, sys 
import numpy as np

from simbig import infer as Infer

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as Ut
from sbi import inference as Inference

sumstat     = sys.argv[1]
kmax        = float(sys.argv[2]) 
nuis        = (sys.argv[3] == 'True') 
n_models    = int(sys.argv[4])
machine     = sys.argv[5]
restrict    = sys.argv[6]
overwrite   = (sys.argv[7] == 'True')


cuda = torch.cuda.is_available()

seed = 12387
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
device = ("cuda" if cuda else "cpu")

if machine == 'adroit': 
    dat_dir = '/scratch/network/chhahn/simbig/'
elif machine == 'della': 
    dat_dir = '/tigress/chhahn/simbig/'
##################################################################################
# load data 
##################################################################################
version = 'v4'
y_train, x_train = Infer.training_data(sumstat, kmax, 
        machine=machine, nuis=nuis, version=version, 
        restrict=(None if restrict == 'None' else restrict)) 
print('Ntrain=%i' % x_train.shape[0])

y_test0, x_test0 = Infer.testing_data(sumstat, kmax, 'test', 
        machine=machine, version=version) 
y_test1, x_test1 = Infer.testing_data(sumstat, kmax, 'test_fof', 
        machine=machine, version=version)
y_test2, x_test2 = Infer.testing_data(sumstat, kmax, 'test_abacus', 
        machine=machine, version=version) 

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

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

##################################################################################
# train 20 NPEs and pick the best one  
##################################################################################
for i in range(n_models):
    nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(128), np.log(256)))))
    nblocks = int(np.random.uniform(5, 10))
    arch = '%ix%i' % (nhidden, nblocks)
    print('MAF with nhidden=%i; nblocks=%i' % (nhidden, nblocks))
   
    if int(kmax * 100) % 10 == 0: 
        kmax_str = 'kmax%.1f' % kmax
    else:  
        kmax_str = 'kmax%.2f' % kmax
    fqphi   = os.path.join(dat_dir, 
            'qphi.%s%s%s.%s.%s.%s.pt' % 
            (version, ['', '.nuis'][nuis], ('' if restrict == 'None' else '.%s' % restrict), arch, sumstat, kmax_str))
    if not os.path.isfile(fqphi) or overwrite: 
        anpe = Inference.SNPE(prior=prior,
                              density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
                              device=device, 
                              summary_writer=SummaryWriter('/home/chhahn/projects/simbig/cmass/bin/inference/o/npe.%s.%s' % (sumstat, kmax_str)))

        anpe.append_simulations(
            torch.as_tensor(y_train.astype(np.float32)).to(device),
            torch.as_tensor(x_train.astype(np.float32)).to(device))

        p_theta_x_est = anpe.train()
        
        # save trained NPE  
        qphi    = anpe.build_posterior(p_theta_x_est)
        torch.save(qphi, fqphi)
        
        f = open(fqphi.replace('.pt', '.valid_logprob'), 'w') 
        f.write('%.5e' % anpe._summary['best_validation_log_probs'][0])
        f.close()
    else: 
        qphi = torch.load(fqphi) 

    # run NPE on test data  
    mcmc0, rank0 = Infer.npe_validation(qphi, x_test0, y_test0, device=device)
    #np.save(fqphi.replace('.pt', '.test0.mcmc.npy'), mcmc0)
    np.save(fqphi.replace('.pt', '.test0.rank_thetas.npy'), rank0)

    _, rank1 = Infer.npe_validation(qphi, x_test1, y_test1, device=device)
    np.save(fqphi.replace('.pt', '.test1.rank_thetas.npy'), rank1)

    _, rank2 = Infer.npe_validation(qphi, x_test2, y_test2, device=device)
    np.save(fqphi.replace('.pt', '.test2.rank_thetas.npy'), rank2)
