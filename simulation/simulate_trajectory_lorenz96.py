import os
import shutil

import torch 
import numpy as np

import torch.nn.functional as func

class Simulator_L96():
    
    """A simulator of Lorenz96 trajectories given a particular dynamics.

    Attributes:
        dynamics    numerical scheme integrating the state over time.
        N    size of the system state
        T    time-length of the integration window
    """
    
    def __init__(self,dynamics,T=20,N=16):
        
        self.dynamics=dynamics
        self.T=T
        self.N=N
        
    def sample(self, seed=None):
        
        """Sample a system trajectory
        
        Keyword arguments:
            seed -- random seed initializer
        """
        
        if seed != None:   
            numpy.random.seed(seed)
        
        # state vector X
        X = torch.zeros((self.T, self.N))
        
        # random initial conditions
        ic = np.random.normal(0,1,X.shape[1]).astype(np.float32)
        X[0,:] = torch.Tensor(ic)
        
        # first integration run to reach equilibrium
        for i in range(1,64):
            X[0,:] = self.dynamics.forward(X[0,:])
        
        # second run - final trajectory
        for t in range(self.T-1):
            X[t+1,:] = self.dynamics.forward(X[t,:])
             
        return X
    
    def dataset(self, n_sample, path_save, seed=42):
        
        """Integrates a state X with the defined shallow water dynamics

        Keyword arguments:
            n_sample -- number of trajectory to sample
            path_save -- path to save the dataset
            seed -- random seed initializer
        """
        
        # set seed so that it's always the same dataset generated 
        np.random.seed(seed)
        
        # Directories management         
        if os.path.exists(path_save):
            shutil.rmtree(path_save)
        os.makedirs(path_save)

        for i in range(n_sample):
            X = self.sample()
            
            # save     
            np.save(path_save+'/'+'{0:04}'.format(int(i)),
                    X.numpy())