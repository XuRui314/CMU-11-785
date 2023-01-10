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
            self.Z     = Z
            self.M     = self.running_M
            self.V     = self.running_V
            self.NZ    = (self.Z - self.M) / np.sqrt(self.V + self.eps) # TODO
            self.BZ    = self.BW * self.NZ + self.Bb # TODO

            return self.BZ
            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        
        self.M         = np.mean(Z, axis = 0, keepdims = True) # TODO
        self.V         = np.var(Z, axis = 0, keepdims = True) # TODO
        self.NZ        = (self.Z - self.M ) / np.sqrt(self.V + self.eps) # TODO
        self.BZ        = self.BW * self.NZ + self.Bb # TODO
        
        self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V # TODO
        
        return self.BZ


    # Notation Clarification
    # Z^hat: normalized input data, Z^widetilde: data output from the BN layer
    # dLdBW -> dL/dW, dLdBb -> dL/db
    # dLdNZ -> dL/dZ^hat, dLdBZ -> dL/dZ^widetilde

    def backward(self, dLdBZ):
        
        self.dLdBW  = dLdBZ * self.NZ.sum(axis = 0, keepdims = True) # TODO
        self.dLdBb  = dLdBZ.sum(axis = 0, keepdims = True) # TODO
        
        dLdNZ       = dLdBZ * self.BW # TODO
        dLdV        = -1/2 # TODO
        dLdM        = None # TODO
        
        dLdZ        = dLdNZ * np.sqrt(self.V + self.eps) + dLdV * (2/self.N * (self.Z - self.M)) + \
         1 / self.N * dLdM  # TODO
        
        return  dLdZ