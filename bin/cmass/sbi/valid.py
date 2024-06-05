import os, sys, glob
import numpy as np
from tqdm.auto import trange

from simbig import infer as Infer


import torch
from sbi import utils as Ut
from sbi import inference as Inference

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

version = 'v4'
if 'della' in os.environ['HOSTNAME']: 
    machine = 'della'
    dat_dir = '/tigress/chhahn/simbig/'
elif 'adroit' in os.environ['HOSTNAME']:
    machine = 'adroit'
    dat_dir = '/scratch/network/chhahn/simbig/'


def run_ensemble(sumstat, kmax, test_set): 
    '''
    '''
    # load ensemble 
    qphis = load_ensemble(sumstat, kmax)    

    # load test set 
    if test_set == 'extra': 
        y_train, x_train = Infer.training_data(sumstat, kmax, machine=machine,
        nuis=False, version=version, restrict=None, extra=True)

        y_test = y_train[20000:]
        x_test = x_train[20000:]
    if test_set == 'train': 
        y_train, x_train = Infer.training_data(sumstat, kmax, machine=machine,
        nuis=False, version=version, restrict=None, extra=True)

        y_test = y_train[:20000][::10]
        x_test = x_train[:20000][::10]
    else: 
        y_test, x_test = Infer.testing_data(sumstat, kmax, test_set, machine=machine,
                version=version)
    
    # deploy NDEs on test set 
    t = trange(y_test.shape[0])

    mcmcs = []
    for ii in t: 
        _y = y_test[ii]
        _x = x_test[ii]
        
        # sample each of the NDEs
        _mcmcs = []
        for qphi in qphis: 
            _mcmc = qphi.sample((2000,), x=torch.as_tensor(_x.astype(np.float32)).to(device), 
                                show_progress_bars=False)
            _mcmc = np.array(_mcmc.detach().to('cpu'))
            _mcmcs.append(_mcmc)

        mcmcs.append(np.concatenate(_mcmcs, axis=0))
    
    if kmax * 100 % 10 == 0: 
        kmax_str = '%.1f' % kmax
    else: 
        kmax_str = '%.2f' % kmax
    study_name = 'qphi.exp1.maf.v4.%s.kmax%s' % (sumstat, kmax_str)
    np.save(os.path.join(dat_dir, 'npe/%s/%s.%s.mcmc.npy' % (study_name, study_name, test_set)), 
            np.array(mcmcs))
    return None 


def load_ensemble(sumstat, kmax): 
    if kmax * 100 % 10 == 0: 
        kmax_str = '%.1f' % kmax
    else: 
        kmax_str = '%.2f' % kmax
    study_name = 'qphi.exp1.maf.v4.%s.kmax%s' % (sumstat, kmax_str)
    fevents = glob.glob(os.path.join(dat_dir, 'npe/%s/*/events*' % study_name))
    if len(fevents) == 0: raise ValueError
        
    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)
    best_valid = np.array(best_valid)

    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i in np.argsort(best_valid)[-5:][::-1]]
    print(i_models) 

    qphis = []
    for i_model in i_models:
        fqphi = os.path.join(dat_dir, 'npe/qphi.exp1.maf.v4.%s.kmax%s/qphi.exp1.maf.v4.%s.kmax%s.%i.pt' %
                 (sumstat, kmax_str, sumstat, kmax_str, i_model))
        qphi = torch.load(fqphi, map_location=device)
        qphis.append(qphi)
    return qphis 


if __name__=="__main__": 
    if sys.argv[1] == 'run_ensemble': 
        sumstat     = sys.argv[2] 
        kmax        = float(sys.argv[3])
        test_set    = sys.argv[4]

        run_ensemble(sumstat, kmax, test_set)
