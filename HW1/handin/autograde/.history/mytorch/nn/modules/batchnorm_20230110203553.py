import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            # TODO
            return self.running_M, self.running_V
            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        
        self.M         = np.mean(Z, axis = 0, keepdims = True) # TODO
        self.V         = np.var(Z, axis = 0, keepdims = True) # TODO
        self.NZ        = None # TODO
        self.BZ        = None # TODO
        
        self.running_M = self.M # TODO
        self.running_V = np.var(Z, axis = 0, keepdims = True) # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = None # TODO
        self.dLdBb  = None # TODO
        
        dLdNZ       = None # TODO
        dLdV        = None # TODO
        dLdM        = None # TODO
        
        dLdZ        = None # TODO
        
        return  NotImplemented