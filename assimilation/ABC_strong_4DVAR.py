import torch
import torch.nn as nn
import torch.optim as optim

class ABC_strong_4DVar():
    
    """Approximate Bayesian Computation using 4DVAR loss function as criteria

    Attributes:
        dynamics                 dynamical model used in the forward operator
        H                        observation operator used in the forward operator
        N_trial                  # of trial
        percent_select_abc       percentage of best trial to keep (posterior)
        percent_select_prior     percentage of random trials to keep (prior)
    """
    
    def __init__(self, dynamics, H,
                 N_trial=10000, percent_select_abc = 1, percent_select_prior = 10):
        
        # Forward model component
        self.dynamics = dynamics
        self.H = H
     
        # init
        self.N_trial = N_trial
        self.size_sample_prior = int(N_trial*percent_select_prior/100)
        self.size_sample_abc = int(N_trial*percent_select_abc/100)

        self.losses_prior=torch.zeros(self.size_sample_prior)
        self.losses_abc = torch.normal(0,1,[self.size_sample_abc])
        self.losses=[]

        # todo or todelete?
        #self.regul=regul
        #self.ground_truth=ground_truth
        #self.monitor=monitor
        
    def Forward(self, X0):
        
        """Forward model model, composition (H) o (dynamics)
        F(X)=Y+eps
        
        Keyword arguments:
            X0 -- initial condition (window beginning)
        """
        
        # Initialize state
        X = torch.zeros(self.Rm1.shape)
        X[0] = X0
        
        # Time integration with dynamics   
        for t in range(1,self.T):
            X[t] = self.dynamics.forward(X[(t-1)].clone())
        
        # Obseravation through H
        Y = self.H.forward(X)
        
        return X, Y
        
    def J_obs(self, Y_hat, Y, Rm1):
        
        """Calculate Observational loss
        
        Keyword arguments:
            Y_hat -- Estimation at observational points
            Y -- Obsevations
            Rm1 -- Variances of observations errors
        """
            
        # Quadratic observational error
        jo = ((Y_hat-Y)*Rm1*(Y_hat-Y)).sum()
        
        return jo
    
    def J_background(self, X0, Xb, Bm1):
        
        """Calculate Background loss
        
        Keyword arguments:
            X0 -- Estimated initial condition 
            Xb -- Background
            Bm1 -- Variances of background errors
        """
        
        if self.Bm1==None:
            jb=0
          
        else:
            jb=((X0-Xb)*Bm1*(X0-Xb)).sum()
    
        return jb

    def fit(self, Y, Rm1, Xb=None, Bm1=None):
        
        """Sample and select  
        
        Keyword arguments:
            Y -- Observations
            Rm1 -- normalized inverse variance of observational errors
            Xb -- Background
            Bm1 -- Variances of background errors
        """
        # Tensor storing results
        X_prior = torch.zeros((self.size_sample_prior, Rm1.shape[0], Rm1.shape[1]))
        X_abc = torch.zeros((self.size_sample_abc, Rm1.shape[0], Rm1.shape[1]))
        
        # Background (optional) and variances
        if (Xb == None) or (Bm1==None):
            self.Xb = torch.zeros(Rm1[0].shape)
            self.Xb = Y[0]
            
            self.Bm1 = torch.zeros(Rm1[0].shape)
            self.Bm1 = Rm1[0]
            
        else:
            self.Xb = Xb
            self.Bm1=Bm1
        
        # Observations and variances
        self.Y=Y
        self.Rm1 = Rm1
        
        # Dimension
        self.T = Rm1.shape[0]
        
        # calculate sigma_b from variance
        sigma_b = (1/Bm1)
        sigma_b[sigma_b==float('inf')]=0
        sigma_b=sigma_b.sqrt()
        
        for n in range(self.N_trial):
            
            noise_b=torch.normal(0, sigma_b)
            X0 = Xb + noise_b
            
            X_hat, Y_hat = self.Forward(X0)

            loss = self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X_hat[0], self.Xb, self.Bm1)
            loss=loss.detach()
            
            if loss != loss:  
                print('Nan loss')

            # periodically add to prior
            if n%int(self.N_trial/self.size_sample_prior)==0:
                X_prior[int(n*self.size_sample_prior/self.N_trial)]=X_hat
                self.losses_prior[int(n*self.size_sample_prior/self.N_trial)]=loss

            # ABC
            if (n < self.size_sample_abc):
                X_abc[n] = X_hat
                self.losses_abc[n]=loss

            if (loss < self.losses_abc.max()):

                replace_index=torch.where(self.losses_abc==self.losses_abc.max())

                X_abc[replace_index] = X_hat
                self.losses_abc[replace_index]=loss

            # store all losses
            self.losses.append(loss)
            
        self.X_prior = X_prior
        self.X_abc = X_abc