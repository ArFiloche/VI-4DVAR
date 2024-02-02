import torch
import torch.nn as nn
import torch.optim as optim

class strong_4DVar():
    
    """Variational Assimilation.

    Attributes:
        dynamics        dynamical model used in the forward operator
        H               observation operator used in the forward operator
        regul           additional regularization (option)
        lr              optimizer (l-BFGS) learning rate
        max_iter        optimizer (l-BFGS) maximum iteration
    """
    
    def __init__(self, dynamics, H,
                 lr=0.1, max_iter=1500, regul=None):
        
        # Forward model component
        self.dynamics = dynamics
        self.H = H
        self.regul=regul
        
        # Optimizer
        self.max_iter = max_iter
        self.lr = lr
        self.optimizer = optim.LBFGS([torch.zeros(0)],
                                     lr=self.lr, max_iter=self.max_iter)
                         #tolerance_grad=0, tolerance_change=0)
       
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
        
        # normalizing constant for numerical optimization
        
        return X, Y
        
    def J_obs(self, Y_hat, Y, Rm1):
        
        """Calculate Observational loss
        
        Keyword arguments:
            Y_hat -- Estimation at observational points
            Y -- Obsevations
            Rm1 -- Variances of observations errors
        """
            
        # Quadratic observational error
        jo = 0.5 * ((Y_hat-Y)*Rm1*(Y_hat-Y)).sum()
    
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

    def fit(self, Y, Rm1, Xb=None, Bm1=None):
        
        """Optimize control parameters on observations 
        
        Keyword arguments:
            Y -- Observations
            Rm1 -- normalized inverse variance of observational errors
        """
        # Monitoring
        self.n_iter = 0
        self.losses=torch.zeros(self.max_iter)*float('Nan')
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

        # eps_b, control paramaters
        self.eps_b = torch.zeros(self.Xb.shape)
        self.eps_b.requires_grad = True
        self.optimizer.param_groups[0]['params'][0] = self.eps_b
        
        #normalizing constant for the numerical optimization
        X0 = self.Xb + self.eps_b
        X_hat, Y_hat = self.Forward(X0.detach())
        initial_loss = self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X0, self.Xb, self.Bm1)
        
        self.K = initial_loss.detach()
        
        def closure():
            
            self.optimizer.zero_grad()
            X0 = self.Xb + self.eps_b
            
            # check for NaN
            if torch.isnan(X0.mean()).item() != 0:          
                print('Nan X0: failed to converge')
                self.convergence=0
                loss = torch.zeros(1,requires_grad = True)
            
            else:
                X_hat, Y_hat = self.Forward(X0)

                # Full state // initial condition
                self.X_hat = X_hat.detach()
                self.initial_condition = self.X_hat[-1].detach()
                
                loss = self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X0, self.Xb, self.Bm1)
                loss = loss/self.K
                
                # check for regularization
                #if self.regul == None:
                #else:
                    #loss  = +self.regul.J(X_hat)
                
                # check for NaN
                if torch.isnan(loss).item() != 0:          
                    print('Nan loss: failed to converge')
                    self.convergence=0
                    loss = torch.zeros(1,requires_grad = True)

                loss.backward(retain_graph=True)

                # Monitor
                self.losses[self.n_iter]=loss.item()
                self.n_iter = self.n_iter + 1

            return loss
        
        loss = self.optimizer.step(closure)
        
        if self.convergence==1:
            # Full state
            X_hat, _ = self.Forward(self.Xb + self.eps_b)
            self.X_hat = X_hat.detach()

            # Initial condition
            self.initial_condition = self.X_hat[-1].detach()
            
    def get_precision_matrix(self):
        
        """Compute precision matrix"""
        
        def J(X0):
            
            _, Y_hat = self.Forward(X0)
            
            j= self.J_obs(Y_hat, self.Y, self.Rm1) + self.J_background(X0, self.Xb, self.Bm1)
              
            return j
        
        # compute Hessian
        self.Hess = torch.autograd.functional.hessian(J, self.X_hat[0])
        
        # inverse Hessian and get precision matrix
        self.Pa = torch.linalg.inv(self.Hess)
        
        return self.Pa