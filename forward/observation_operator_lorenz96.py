import numpy as np
import torch
import torch.nn as nn

class Observation_operator_L96:
    
    """A class of observation operator for shallow water trajectory.

    Attributes:
        r     downscaling factor
        subsample_t     time subsampling factor
        p      proba of observing each component at each time
        nonlin non-linear observation operator (optional)
    """
    
    def __init__(self,r=1, subsample_t=1, p=0.5, nonlin=None):
        
        self.r=r
        self.subsample_t=subsample_t
        self.proba_drop = [1-p, p]
        self.drop_value = [1, 0]#[1, float('nan')]
        
        # linear projection drop
        self.drop = np.random.choice(self.drop_value, (100,100),
                                p=self.proba_drop).astype(np.float32)
        self.drop = torch.Tensor(self.drop)
        self.drop[0,:]=0
        
        self.nonlin = nonlin
        
    def forward(self, X):
        
        """Observes a state X.

        Keyword arguments:
            X -- system state at previous time-step
        """
    
        T, N = X.shape
        drop = self.drop[0:T,0:N]
        Y = torch.zeros(X.shape)

        # decimation operator for spatial downscaling
        decimation= nn.AvgPool1d(self.r, stride=self.r, padding=0)
    
        for t in range(T):
            if (t%self.subsample_t==0):
                x_t = X[t,:].unsqueeze(0).unsqueeze(0)
                Y[t,:]=decimation(x_t).squeeze(0).squeeze(0)
        
        if self.nonlin == None:
            Y = drop*Y
        else:
            Y = drop*self.nonlin(Y)
    
        return Y 