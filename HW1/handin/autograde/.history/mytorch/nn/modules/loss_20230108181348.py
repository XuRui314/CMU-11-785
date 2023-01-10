import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = (Y - A) ** 2 # TODO
        sse    = se.sum() # TODO
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = 2 * (self.A - self.Y) / self.A.shape[0] * self.A.shape[1]
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.exp(A) / np.exp(A).sum(axis = 1) # TODO
        crossentropy     = Y * np.log(self.softmax) # TODO
        sum_crossentropy = crossentropy.sum() # TODO
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = None # TODO
        
        return NotImplemented
