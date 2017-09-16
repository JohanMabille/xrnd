import numpy as np
import timeit

def GrayScott(counts, Du, Dv, F, k):
    n = 300
    U = np.zeros((n+2,n+2))
    V = np.zeros((n+2,n+2))
    u, v = U[1:-1,1:-1], V[1:-1,1:-1]

    r = 20
    u[:] = 1.0
    nd2 = int(n/2)
    U[nd2-r:nd2+r,nd2-r:nd2+r] = 0.50
    V[nd2-r:nd2+r,nd2-r:nd2+r] = 0.25
    u += 0.15*np.ones((n,n))
    v += 0.15*np.ones((n,n))

    uvv = np.empty_like(u)
    for i in range(counts):
        Lu = (                 U[0:-2,1:-1] +
              U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                               U[2:  ,1:-1] )
        Lv = (                 V[0:-2,1:-1] +
              V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                               V[2:  ,1:-1] )
        uvv[:] = u*v*v
        u += Du*Lu - uvv + F*(1 - u)
        v += Dv*Lv + uvv - (F + k)*v

    return V

def test():
    GrayScott(40, 0.16, 0.08, 0.04, 0.06)

print(timeit.timeit("test()", setup = "from __main__ import test", number=100))
