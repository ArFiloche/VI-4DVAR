import torch
import torch.nn as nn
import torch.optim as optim

from .strong_4DVar import strong_4DVar

class Ensemble_strong_4DVar():
    
    """Ensemble of 4D Variational Assimilation
       adapted from [Jardak & Tallagrand, 2018] https://npg.copernicus.org/articles/25/589/2018/

    Attributes:
        dynamics        dynamical model used in the forward operator
        H               observation operator used in the forward operator
        N_member        # of members in the ensemble
        regul           additional regularization (option)
        lr              optimizer (l-BFGS) learning rate
        max_iter        optimizer (l-BFGS) maximum iteration
    """
    
    def __init__(self, dynamics, H, N_member=100,
                 lr=0.1, max_iter=1500, regul=None):
        
        # Ensemble
        self.N_member=N_member
        self.Elosses=torch.zeros((N_member,max_iter))
        self.En_iter=torch.zeros(N_member)
        
        # 4DVAR
        self.assimilation=strong_4DVar(dynamics, H,
                                       lr, max_iter,regul)

    def fit(self, Y, Rm1, Xb=None, Bm1=None):
        
        """Optimize control parameters on observations 
        
        Keyword arguments:
            Y -- Observations
            Rm1 -- normalized inverse variance of observational errors
            Xb -- Background
            Bm1 -- Variances of background errors
        """
        # store ensemble results
        self.Ensemble=torch.zeros(self.N_member, Rm1.shape[0], Rm1.shape[1])
        
        # Background (optional) and variances
        if (Xb == None) or (Bm1==None):
            self.Xb = torch.zeros(Rm1[0].shape)
            self.Xb = Y[0]
            
            self.Bm1 = torch.zeros(Rm1[0].shape)
            self.Bm1 = Rm1[0]
            
        else:
            self.Xb = Xb
            self.Bm1=Bm1
        
        # calculate sigma_b from variance
        sigma_b = (1/Bm1)
        sigma_b[sigma_b==float('inf')]=0
        sigma_b=sigma_b.sqrt()
        
        # Observations and variances
        self.Y=Y
        self.Rm1 = Rm1
        
        # calculate sigma_obs from variance
        sigma_obs = (1/Rm1)
        sigma_obs[sigma_obs==float('inf')]=0
        sigma_obs=sigma_obs.sqrt()
        
        # loop for N_member 4DVAR
        for n in range(self.N_member):
            
            noise_b=torch.normal(0, sigma_b)
            noise_obs=torch.normal(0, sigma_obs)
            
            Xb_n = Xb+noise_b
            Y_n = Y+noise_obs
            
            self.assimilation.fit(Y_n, Rm1, Xb_n, Bm1)
            
            self.Ensemble[n]=self.assimilation.X_hat
            self.Elosses[n]=self.assimilation.losses
            self.En_iter[n]=self.assimilation.n_iter