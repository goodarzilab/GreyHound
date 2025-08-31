import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import numpy as np

class vaeDataset(Dataset):
    def __init__(self, RBP_df, exp_df):
        self.RBP_df = RBP_df
        self.exp_df = exp_df
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (torch.from_numpy(self.RBP_df[idx].astype(np.float32)),
                torch.from_numpy(self.exp_df[idx].astype(np.float32)))
    
    def __len__(self):
        return self.exp_df.shape[0]

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size*10),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size*10))
        
        self.fc_mu = nn.Linear(latent_size*10, latent_size)
        self.fc_var = nn.Linear(latent_size*10, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size*10),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size*10),
            nn.Linear(latent_size*10, output_size))
        
    def encode(self, input):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return [mu, log_var]
    
    def decode(self, z):
        out = self.decoder(z)
        return out
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]
    
    def loss_fnc(self,recons, input, mu, log_var, kld_weight=0.5):
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #print("recon_loss: {}, kld_loss:{}".format(recons_loss, kld_loss))
        return recons_loss + kld_weight * kld_loss, recons_loss, kld_loss
    
    def compile(self, opt="Adam", loss="entropy", learning_rate=0.001, device='cpu'):
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def train_model(self, train_loader, epochs=20, device='cpu'):
        beta = 0 #kld_weight
        step = 0

        train_losses = []
        recon_losses = []
        kld_losses = []
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            recon_loss = 0.0
            kld_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                input, target = batch
                input = input.to(device)
                target= target.to(device)
                recon, mu, logvar = self(input)
                
                ##Warmp up
                beta = np.log(1+step) / 500*(1+np.log(1+step))
                step += 1
                
                loss, loss_rec, loss_kld = self.loss_fnc(recon, target, mu, logvar, kld_weight=beta)
                
                loss.backward()
                train_loss += loss.data
                recon_loss += loss_rec.data
                kld_loss += loss_kld.data
                
                self.optimizer.step()
                
                if step % 100 == 0:
                    train_losses.append(train_loss)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon Loss: {:.6f}\tKL Loss: {:.6f}'.format(
                            epoch, batch_idx * len(input), (len(train_loader)*128),
                            100. * batch_idx / len(train_loader),
                            loss.data / len(input), loss_rec.data/ len(input), loss_kld.data/ len(input)))
            
            train_losses.append (train_loss / len(train_loader.dataset))
            recon_losses.append (recon_loss / len(train_loader.dataset))
            kld_losses.append (kld_loss / len(train_loader.dataset))
            
        return {'train_loss':train_losses, 'recon_losses':recon_losses, 'kld_losses':kld_losses}
        
    def encode_latent(self, input):
        model.eval()
        input.to(device)
        recon, mu, logvar = model(input)
        return mu
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
