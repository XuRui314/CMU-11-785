import numpy as np
# N is the num of samples, C is the dimension
# Z : N x C
# A : N x C

class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
    
        self.A = 1 / ( 1 + np.exp(-Z)) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A - self.A * self.A  # TODO
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.tanh(Z) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - self.A * self.A # TODO
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = np.max(Z, 0) # TODO
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.where(self.A # TODO
        
        return NotImplemented
        
        
