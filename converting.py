import numpy as np

def up_hat(p):          # the ^ calculation
    P = np.zeros([3,3])
    P[0,1] = -p[2]
    P[0,2] = p[1]
    P[1,0] = p[2]
    P[1,2] = -p[0]
    P[2,0] = -p[1]
    P[2,1] = p[0]
    return P