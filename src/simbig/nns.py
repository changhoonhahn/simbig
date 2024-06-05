'''

module for some NNs that we'll be using  


'''
import numpy as np 
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import trange


class MLP(nn.Module):
    ''' simple MLP used for a quick compression scheme. 
    '''
    def __init__(self, input_size, output_size, nhidden, final_tanh=False, batch_norm=False, dropout_prob=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if isinstance(nhidden, int):nhidden = [nhidden]
        self.nlayers = len(nhidden)

        layers = []
        dropout = dropout_prob != 0.0
        if batch_norm:
            layers += [nn.BatchNorm1d(num_features=self.input_size, track_running_stats=False)]
        layers += [nn.Linear(self.input_size, nhidden[0]), nn.ReLU()]
        if dropout:
            layers += [nn.Dropout(dropout_prob)]
        for i in range(self.nlayers - 1):
            if batch_norm:
                layers += [nn.BatchNorm1d(num_features=nhidden[i], track_running_stats=False)]
            layers += [nn.Linear(nhidden[i], nhidden[i+1]), nn.ReLU()]
            if dropout:
                layers += [nn.Dropout(dropout_prob)]
        layers += [nn.Linear(nhidden[-1], self.output_size)]
        if final_tanh:
            layers += [nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def TrainMLP(mlp, train_loader, valid_loader, lrate=1e-3, epochs=1000,
        patience=20, device=None, optimizer=None, scheduler=None, save_path=None,
        weight_decay=0):
    ''' train MLP above with adam optimizer and onecycle learning rate
    '''
    if optimizer is None:
        optimizer = optim.Adam(mlp.parameters(), lr=lrate, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lrate, epochs=epochs, steps_per_epoch=len(train_loader))

    best_epoch = 0
    best_valid_loss = np.inf
    train_losses, valid_losses = [], []
    t = trange(epochs, leave=False)
    for epoch in t:
        mlp.train()
        train_loss = 0
        for batch in train_loader:
            _x, _y = batch
            _x = _x.to(device)
            _y = _y.to(device)
            optimizer.zero_grad()

            loss = torch.sqrt(torch.sum((_y - mlp.forward(_x))**2, dim=-1))
            loss = torch.mean(loss)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        with torch.no_grad():
            valid_loss = 0
            for val_batch in valid_loader:
                x_val, y_val = val_batch
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                _loss = torch.sqrt(torch.sum((y_val - mlp.forward(x_val))**2, dim=-1))
                _loss = torch.mean(_loss)
                valid_loss += _loss.item()
            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

        scheduler.step()

        t.set_description('Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss), refresh=False)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_model = copy.deepcopy(mlp)
            if save_path is not None:
                torch.save(mlp.state_dict(), save_path)
        else:
            if epoch > best_epoch + patience:
                print('Stopping')
                print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, best_valid_loss))
                break

    if save_path is not None:
        torch.save(mlp.state_dict(), save_path)

    return best_model, best_valid_loss, train_losses, valid_losses
