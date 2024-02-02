import torch
import torch.nn as nn
import torch.optim as optim

class VI_strong_4DVar():
    
    """Variational Inference 4DVAR

    Attributes:
        dynamics        dynamical model used in the forward operator
        H               observation operator used in the forward operator
        lr              optimizer (Adam) learning rate
        N_iter          optimizer number of iteration
        batch_size      # of sample for the Monte Carlo estimation of the gradient
        N_final_sample  # of sample to represent the estimated posterior after optimization
        
    """
    
    def __init__(self, dynamics, H,
                 lr=0.01, N_iter=1500, batch_size=1,
                 N_final_sample=1000):
        
        # Forward model component
        self.dynamics = dynamics
        self.H = H
        
        # Optimizer
        self.N_iter = N_iter
        self.batch_size=batch_size
        self.lr = lr
        self.optimizer = optim.Adam([torch.zeros(0)],
                                    lr=self.lr, betas=(0.9, 0.999))
        
        self.N_final_sample=N_final_sample
    
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
        jo = 0.5*((Y_hat-Y)*Rm1*(Y_hat-Y)).sum()
        
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
            jb=0.5*((X0-Xb)*Bm1*(X0-Xb)).sum()
    
        return jb
    
    def gaussian_ll(self,X0, mean, log_var, eps: float = -3):
    
        # Clamp for stability
        log_var = log_var.clone()
        
        with torch.no_grad():
            log_var.clamp_(min=eps)  
            
        var = torch.exp(log_var)
        
        # compute loss
        nll = 0.5*(log_var + (X0 - mean)**2 / var)

        return -nll.sum()

    def fit(self, Y, Rm1, Xb=None, Bm1=None):
        
        # Monitoring
        self.n_iter = 0
        
        # monitoring losses
        self.losses=torch.zeros(self.N_iter)*float('Nan')
        self.losses_nll = torch.zeros(self.N_iter)*float('Nan')
        self.losses_4dvar = torch.zeros(self.N_iter)*float('Nan')
        
        self.convergence=1
        
        # Background (optional) and variances
        if Xb == None:
            self.Xb = torch.zeros(Rm1[0].shape)
            self.Xb = Y[0]           
        else:
            self.Xb = Xb
            
        self.Bm1=Bm1
        
        # Observations and variances
        self.Y=Y
        self.Rm1 = Rm1
        
        # Dimension
        self.T = Rm1.shape[0]
        
        # control paramaters
        self.params = torch.zeros(list([2])+list(Xb.shape))  # initial mu = 0
        self.params[1] = self.params[1] # initial log_var = 0
        self.params.requires_grad = True
        self.optimizer.param_groups[0]['params'][0] = self.params
        
        ##normalizing constant for the numerical optimization      
        X0 = self.Xb #+ mu
        X_hat, Y_hat = self.Forward(X0.detach())
        initial_loss_4dv = self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X0, self.Xb, self.Bm1)
        self.K = initial_loss_4dv.detach() #normalization constant
        
        for n_iter in range(self.N_iter):
            
            self.optimizer.zero_grad()
            loss = torch.zeros(1,requires_grad = True)
            
            mu = self.params[0]
            log_var = self.params[1]
        
            # Clamp for stability
            log_var = log_var.clone()
            with torch.no_grad():
                log_var.clamp_(-3)
               
            for batch in range(self.batch_size):
                
                X0 = self.Xb+mu
                X0 = X0 + torch.randn_like(log_var)*torch.exp(0.5*log_var)
            
                # check for NaN / convergence issue
                if torch.isnan(X0.mean()).item() != 0:          
                    print('Nan X0: failed to converge')
                    self.convergence=0
                    loss = torch.zeros(1,requires_grad = True)

                else:
                    X_hat, Y_hat = self.Forward(X0)

                    loss_nll  = self.gaussian_ll(X0, mu, log_var)
                    loss_4dvar = self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X0, self.Xb, self.Bm1)
                    loss = loss_nll + loss_4dvar
                    
                    # check for NaN / convergence issue
                    if torch.isnan(loss).item() != 0:          
                        print('Nan loss: failed to converge')
                        self.convergence=0
                        loss = torch.zeros(1,requires_grad = True)
                        
            loss = loss/(self.batch_size*self.K)
            
            # Monitor
            self.losses[self.n_iter]=loss.item()
            self.losses_nll[self.n_iter]=loss_nll.item()/(self.batch_size*self.K)
            self.losses_4dvar[self.n_iter]=loss_4dvar.item()/(self.batch_size*self.K)

            # Backpropagate
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            self.n_iter = self.n_iter + 1
            
        # store result
        self.mean = self.Xb+self.params.detach()[0]
        self.log_var = self.params.detach()[1]
        self.var = torch.exp(self.log_var)
        self.std = self.var.sqrt()
        
        self.X_vi=torch.zeros(self.N_final_sample, self.Rm1.shape[0], self.Rm1.shape[1])
            
        for n in range(self.N_final_sample):   
            X0 = self.mean+torch.randn_like(self.std)*self.std
            X_hat, Y_hat = self.Forward(X0)
            self.X_vi[n]=X_hat       