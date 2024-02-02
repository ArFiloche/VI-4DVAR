import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# seed set before sampling noise - lign 66

class Simulator_obs():
    
    def __init__(self, H, sigma_perc_b=[15,25], sigma_perc_obs=[5,10]):
        
        self.H = H
        self.sigma_perc_b = sigma_perc_b
        self.sigma_perc_obs = sigma_perc_obs
    
    def sample(self, X, seed=None):
        
        if seed != None:   
            torch.manual_seed(seed)
            
        ### Observation ###
        
        #apply observation operator                                                 
        Y = self.H.forward(X)
        
        # Mask / normalized inverse variance
        Rm1_mask=torch.zeros(Y.shape)
        mask = self.H.forward(torch.ones(X.shape))
        Rm1_mask[mask!=0]=1
        
        # Sample observational noise
        # uniform sample the interval
        sigma_p_o = np.random.uniform(self.sigma_perc_obs[0], self.sigma_perc_obs[1], X.shape)
        sigma_p_o = torch.Tensor(sigma_p_o)
        # calculate signal amplitude
        obs_cc = Y[Rm1_mask!=0].max()-Y[Rm1_mask!=0].min()
        # sigma is a percentage of amplitude
        sigma_obs = sigma_p_o*((obs_cc/2)/100)
        sigma_obs =  sigma_obs*Rm1_mask

        noise_obs=torch.normal(0, sigma_obs)
        #noise_obs = noise_obs*Rm1_mask
        
        Y = Y + noise_obs
        
        ### Background ###
        
        # fully observed 
        Bm1_mask = torch.ones(X[0].shape)
        
        # Sample background noise
        # uniform sample the interval
        sigma_p_b = np.random.uniform(self.sigma_perc_b[0], self.sigma_perc_b[1], X[0].shape)
        sigma_p_b = torch.Tensor(sigma_p_b)
        # calculate signal amplitude
        b_cc = X[0][Bm1_mask!=0].max()-X[0][Bm1_mask!=0].min()
        # sigma is a percentage of amplitude
        sigma_b = sigma_p_b*((b_cc/2)/100)
        sigma_b = sigma_b*Bm1_mask
        
        noise_b=torch.normal(0, sigma_b)
        #noise_b=noise_b*Bm1_mask
        
        Xb = X[0]+noise_b       
    
        return Xb, sigma_b, Bm1_mask, Y, sigma_obs, Rm1_mask
    
    def dataset(self, path_read, path_save, seed=42):
        
        # Directories management         
        if os.path.exists(path_save):
            shutil.rmtree(path_save)
        os.makedirs(path_save)
        
        # read - generate obs - save
        for i in range(len(os.listdir(path_read))):
            X = torch.Tensor(np.load(path_read+'/'+'{0:04}'.format(int(i))+'.npy'))
            Y, Rm1,_ = self.sample(X, seed=seed)  
            np.save(path_save+'/'+'{0:04}'.format(int(i)),
                    torch.stack([Y,Rm1]).numpy())