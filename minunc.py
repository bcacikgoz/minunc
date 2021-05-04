import numpy as np


def is_nonpos_def(x):
    return np.all(np.linalg.eigvals(x) <= 0)
def is_singular(x):
    return np.isinf(np.linalg.cond(x))

def MatthewDavies(H):
    n = H.shape
    n = n[0]
    H = np.hstack((np.zeros((n, 1)), H))
    H = np.vstack((np.zeros((1,n+1)),H))
    nb = H.shape
    nb = nb[0]
    D = np.zeros((nb,nb))
    L = np.zeros((nb,nb))
    if H[1,1]>0:
        H[0,0] = H[1,1]
    else:
        H[0,0] = 1
    for k in range(2,n+1):
        m = k-1
        L[m,m] = 1
        if H[m,m]<=0:
            H[m,m] = H[0,0]
        for i in range(k,n+1):
            L[i,m] = -H[i,m]/H[m,m]
            H[i,m] = 0
            for j in range(k,n+1):
                H[i,j] = H[i,j] + L[i,m]*H[m,j]
        if H[k,k]>0 and H[k,k]<H[0,0]:
            H[0,0] = H[k,k]
    L[n,n] = 1
    if H[n,n]<=0:
        H[n,n] = H[0,0]
    for i in range(1,n+1):
        D[i,i]=H[i,i]
    L = np.delete(L,0,0)
    L = np.delete(L,0,1)
    D = np.delete(D,0,0)
    D = np.delete(D,0,1)
    H = np.delete(H,0,0)
    H = np.delete(H,0,1)
    return L,D


class optimOptions:
    def __init__(self,  gradTolerance = 1e-6, maxIter = 100,
                 armijoLineSearchAlpha = 1, armijoLineSearchDelta = 0.05, armijoLineSearchBeta = 0.6,
                 exactLineSearchIntervalTolerance = 1e-6,exactLineSearchLowerBound = 0, exactLineSearchUpperBound = 1,
                GQTnonposdefBeta = 1e3, GQTposdefBeta = 1):
        self.gradTolerance = gradTolerance
        self.maxIter = maxIter
        self.armijoLineSearchAlpha = armijoLineSearchAlpha
        self.armijoLineSearchDelta = armijoLineSearchDelta
        self.armijoLineSearchBeta = armijoLineSearchBeta
        self.exactLineSearchLowerBound = exactLineSearchLowerBound
        self.exactLineSearchUpperBound = exactLineSearchUpperBound
        self.GQTposdefBeta = GQTposdefBeta
        self.GQTnonposBeta = GQTnonposdefBeta
        self.exactLineSearchIntervalTolerance = exactLineSearchIntervalTolerance

def gsSearch(descentCost, optimOptionsInstance):
    L = optimOptionsInstance.exactLineSearchLowerBound
    U = optimOptionsInstance.exactLineSearchUpperBound
    K = 1.618034
    I = [0]
    xa = [0]
    xb = [0]
    fa = [0]
    fb = [0]
    I.append(U-L)
    I.append(I[1]/K)
    xa.append(U - I[2])
    xb.append(L + I[2])
    fa.append(descentCost(xa[1]))
    fb.append(descentCost(xb[1]))
    k = 1
    while 1:
        I.append(I[k+1]/K)
        if fa[k]>fb[k]:
            L = xa[k]
            xa.append(xb[k])
            xb.append(L+I[k+2])
            fa.append(fb[k])
            fb.append(descentCost(xb[k+1]))
        else:
            U = xb[k]
            xa.append(U - I[k+2])
            xb.append(xa[k])
            fb.append(fa[k])
            fa.append(descentCost(xa[k+1]))

        if I[k]<optimOptionsInstance.exactLineSearchIntervalTolerance or xa[k+1] > xb[k+1]:
            if fa[k+1] > fb[k+1]:
                minimizer = (xb[k+1]+U)/2
            elif fa[k+1] == fb[k+1]:
                minimizer = (xa[k+1] + xb[k+1])/2
            else:
                minimizer = (L + xa[k+1])/2
            minimum = descentCost(minimizer)
            return minimizer, minimum
        k = k+1


def btLineSearch(x, grad, direction, cfunc, optimOptionsInstance):
    g = grad
    d = direction
    delta = optimOptionsInstance.armijoLineSearchDelta
    alpha = optimOptionsInstance.armijoLineSearchAlpha
    beta = optimOptionsInstance.armijoLineSearchBeta
    costFunc = cfunc
    while 1:
        decVal = costFunc(x+alpha*d)
        upLimVal = costFunc(x) + delta*alpha*np.real(np.vdot(np.conj(g), d))
        if decVal>upLimVal:
            alpha = alpha*beta
        else:
            return alpha



def steepestDescent(init, cfunc, gfunc, optimOptionsInstance):
    x = init
    while 1:
        g = gfunc(x)
        d = -g
        descentCost = lambda alpha: cfunc(x + alpha*d)
        ss = gsSearch(descentCost, optimOptionsInstance)
        ss = ss[0]
        xnext = x + ss*d
        fnext = cfunc(xnext)
        if np.linalg.norm(ss*d) < optimOptionsInstance.gradTolerance:
            minimizer = xnext
            minimum = fnext
            return minimizer, minimum
        x = xnext

def spectralGradient(init, cfunc, gfunc, optimOptionsInstance):
    maxIter = optimOptionsInstance.maxIter
    tol = optimOptionsInstance.gradTolerance
    x = init
    xprev = init
    k = 1
    while k<maxIter:
        gprev = gfunc(xprev)
        g = gfunc(x)
        delx = x - xprev
        delg = g - gprev
        if (np.linalg.norm(delx)**2)==0 and (np.vdot(delg,delx)) ==0:
            alpha = 1;
        else:
            alpha = (np.linalg.norm(delx)**2)/(np.vdot(delg,delx))
        xprev = x
        x = x - alpha*g
        if np.linalg.norm(g)<tol:
            minimizer = x
            minimum = cfunc(x)
            return  minimizer, minimum
        k = k+1

def cgFletcherReeves(init, cfunc, gfunc, optimOptionsInstance):
    maxIter = optimOptionsInstance.maxIter
    tol = optimOptionsInstance.gradTolerance
    x = init
    k = 1
    g = gfunc(x)
    d = -g
    ss = btLineSearch(x, g, d, cfunc, optimOptionsInstance)
    gprev = g
    dprev = d
    x = x + ss*d
    while k<maxIter:
        g = gfunc(x)
        beta = (np.linalg.norm(g)**2)/(np.linalg.norm(gprev)**2)
        d = -g + beta*dprev
        ss = btLineSearch(x,g,d, cfunc, optimOptionsInstance)
        x = x + ss*d
        dprev = d
        gprev = g
        k = k +1
        if np.linalg.norm(g)**2<tol:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum





def cgPRP(init, cfunc, gfunc, optimOptionsInstance):
    maxIter = optimOptionsInstance.maxIter
    tol = optimOptionsInstance.gradTolerance
    x = init
    k = 1
    g = gfunc(x)
    d = -g
    ss = btLineSearch(x, g, d, cfunc, optimOptionsInstance)
    gprev = g
    dprev = d
    x = x + ss*d
    while k<maxIter:
        g = gfunc(x)
        beta = (np.vdot(g, g-gprev))/(np.linalg.norm(gprev)**2)
        d = -g + beta*dprev
        ss = btLineSearch(x,g,d, cfunc, optimOptionsInstance)
        x = x + ss*d
        dprev = d
        gprev = g
        k = k +1
        if np.linalg.norm(g)**2<tol:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum





def cgHF(init, cfunc, gfunc, optimOptionsInstance):
    maxIter = optimOptionsInstance.maxIter
    tol = optimOptionsInstance.gradTolerance
    x = init
    k = 1
    g = gfunc(x)
    d = -g
    ss = btLineSearch(x, g, d, cfunc, optimOptionsInstance)
    gprev = g
    dprev = d
    x = x + ss*d
    while k<maxIter:
        g = gfunc(x)
        beta = (np.vdot(g, g-gprev))/(np.vdot(dprev, (g-gprev)))
        d = -g + beta*dprev
        ss = btLineSearch(x,g,d, cfunc, optimOptionsInstance)
        x = x + ss*d
        dprev = d
        gprev = g
        k = k +1
        if np.linalg.norm(g)**2<tol:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum




def cgCD(init, cfunc, gfunc, optimOptionsInstance):
    maxIter = optimOptionsInstance.maxIter
    tol = optimOptionsInstance.gradTolerance
    x = init
    k = 1
    g = gfunc(x)
    d = -g
    ss = btLineSearch(x, g, d, cfunc, optimOptionsInstance)
    gprev = g
    dprev = d
    x = x + ss*d
    while k<maxIter:
        g = gfunc(x)
        beta = -(np.linalg.norm(g)) / (np.vdot(dprev, (gprev)))
        d = -g + beta*dprev
        ss = btLineSearch(x,g,d, cfunc, optimOptionsInstance)
        x = x + ss*d
        dprev = d
        gprev = g
        k = k +1
        if np.linalg.norm(g)**2<tol:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum

def gqtNewton(init, cfunc, gfunc, hfunc, optimOptionsInstance):
    x =  init
    k = 1
    while k<optimOptionsInstance.maxIter:
        g = gfunc(x)
        H = hfunc(x)
        beta = 0
        while is_singular(H):
            if is_nonpos_def(H):
                beta = optimOptionsInstance.GTQnonposdefBeta
            else:
                beta = optimOptionsInstance.GTQposdefBeta
        H = (H + beta*np.identity(np.size(H,0)))/(1+beta)
        d = -np.matmul(np.linalg.inv(H), g)
        descentCost = lambda alpha: cfunc(x + alpha * d)
        ss = gsSearch(descentCost, optimOptionsInstance)
        ss = ss[0]
        x = x + ss*d
        gnew = gfunc(x)
        if np.linalg.norm(gnew)**2<optimOptionsInstance.gradTolerance:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
        k = k+1
        print(k)
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum


def mdNewton(init, cfunc, gfunc, hfunc, optimOptionsInstance):
    x =  init
    k = 1
    while k<optimOptionsInstance.maxIter:
        g = gfunc(x)
        H = hfunc(x)
        L , H = MatthewDavies(H)
        d = -np.matmul(np.matmul(np.transpose(L),np.linalg.inv(H)), np.matmul(L, g))
        descentCost = lambda alpha: cfunc(x + alpha * d)
        ss = gsSearch(descentCost, optimOptionsInstance)
        ss = ss[0]
        x = x + ss*d
        gnew = gfunc(x)
        if np.linalg.norm(gnew)**2<optimOptionsInstance.gradTolerance:
            minimizer = x
            minimum = cfunc(x)
            return minimizer, minimum
        k = k+1
    minimizer = x
    minimum = cfunc(x)
    return minimizer, minimum

