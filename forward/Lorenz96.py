import torch 
import numpy as np
import torch.nn.functional as func

class RK4L96():
    
    """A class of a Lorenz96 dynamics integrator.

    Attributes:
        dt   integration time-step
        F    forcing constant
        k    normalizing constant
        unsqz  boolean allowing multiple integration at the same time
        device     cpu or cuda
    """
    
    def __init__(self, dt=0.1, F=8, K=1, unsqz=False, device='cpu'):
        
        self.dt = dt
        self.F = F
        self.K = K
        
        self.unsqz = unsqz
        self.device = device
        
    def L96(self, X, t):
        
        """Integrates a state X with the defined Lorenz96 dynamics.

        Keyword arguments:
            X -- system state at previous time-step
        """
    
        if self.unsqz:
            N = X.shape[0]
            
            X_pad = func.pad(X.view(X.shape[0],1,-1), pad=(2,2), mode='circular').squeeze()
            
            Xi = X_pad[2:N+2,:]    
            Xip1 = X_pad[3:N+3,:]
            Xim1 = X_pad[1:N+1,:]
            Xim2 = X_pad[0:N,:]
                
            d = ((Xip1 - Xim2) * Xim1 - Xi + self.F)/self.K
                    
        else:
            N = X.shape[0]
                
            X_pad = func.pad(X.view(1,1,-1), pad=(2,2), mode='circular').squeeze()
                
            Xi = X_pad[2:N+2]    
            Xip1 = X_pad[3:N+3]
            Xim1 = X_pad[1:N+1]
            Xim2 = X_pad[0:N]
                
            d = ((Xip1 - Xim2) * Xim1 - Xi + self.F)/self.K
        
        return d

    def forward(self, X):
        
        """Integrates a state X with a RK4 integration scheme.

        Keyword arguments:
            X -- system state at previous time-step
        """
        
        X_int=torch.zeros(X.shape, device=self.device)
        
        k1 = self.L96(X, t=0)
        k2 = self.L96(X+k1*self.dt/2, t=self.dt/2)
        k3 = self.L96(X+k2*self.dt/2, t=self.dt/2)
        k4 = self.L96(X+k3*self.dt, t=self.dt/2)

        X_int = X + (self.dt/6)*(k1+2*k2+2*k3+k4)

        return X_int