import numpy as np
import minunc

def costFuncVect(x):
    return x[0]**2 + 2*(x[1]**2) + 4*(x[0]) + 4*x[1]

def costFuncGrad(x):
    res = np.ones((2,1))
    res[0] = 2*x[0] + 4
    res[1] = 4*x[1] + 4
    return res

def costFuncHess(x):
    hes = np.ones((2,2))
    hes[0,0] = 2
    hes[0,1] = 0
    hes[1,0] = 0
    hes[1,1] = 4
    return hes

defaultOptions = minunc.optimOptions()

minimizer, minimum = minunc.steepestDescent(2*np.ones((2,1)), costFuncVect, costFuncGrad, defaultOptions)
print(minimizer)

minimizerSpectral, minimumSpectral = minunc.spectralGradient(np.zeros((2,1)), costFuncVect, costFuncGrad, defaultOptions)
print(minimizerSpectral)

cgFletcherReevesMinimizer, cgFletcherReevesMinimum = minunc.cgFletcherReeves(np.zeros((2,1)), costFuncVect, costFuncGrad, defaultOptions)
print(cgFletcherReevesMinimizer)

cgPRPMinimizer, cgPRPMinimum = minunc.cgPRP(np.zeros((2,1)), costFuncVect, costFuncGrad, defaultOptions)
print(cgPRPMinimizer)

cgHFMinimizer, cgHFMinimum = minunc.cgHF(np.zeros((2,1)), costFuncVect, costFuncGrad,defaultOptions)
print(cgHFMinimizer)

cgCDMinimizer, cgCDMinimum = minunc.cgCD(np.zeros((2,1)), costFuncVect, costFuncGrad, defaultOptions)
print(cgCDMinimizer)

minimizergqtNewton, minimumgqtNewton = minunc.gqtNewton(2*np.ones((2,1)), costFuncVect, costFuncGrad, costFuncHess,defaultOptions)
print(minimizergqtNewton)


minimizermdNewton, minimummdNewton = minunc.mdNewton(2*np.ones((2,1)), costFuncVect, costFuncGrad, costFuncHess,defaultOptions)
print(minimizermdNewton)

