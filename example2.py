import numpy as np
import minunc
A = np.array([[1, 0], [1, 1], [1, 2]])
b = np.array([[6], [0], [0]])
def obj(x):
    return np.linalg.norm( A@x-b )**2
def grad(x):
    return (np.transpose(A))@(A@x-b)
x0 = np.zeros((2,1))

defaultOptions = minunc.optimOptions()
x, min = minunc.cgPRP(np.zeros((2,1)), obj, grad, defaultOptions)
print(x)
