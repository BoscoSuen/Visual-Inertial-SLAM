import numpy as np

def get_calibration_matrix(K,b):
    M = np.zeros([4,4])
    M[0:2,0:3] = K[0:2,0:3]
    M[2:4,0:3] = K[0:2,0:3]
    M[2,3] = -K[0,0]*b
    return M

def up_hat(p):          # the hat map calculation
    P = np.zeros([3,3])
    P[0,1] = -p[2]
    P[0,2] = p[1]
    P[1,0] = p[2]
    P[1,2] = -p[0]
    P[2,0] = -p[1]
    P[2,1] = p[0]
    return P

def get_pi(q):         # the pi(q) calculation
    return q/q[2]


def dpi_dq(q):         # the dpi/dq calculation
    dpi_dq = np.eye(4)
    dpi_dq[0,2] = -q[0]/q[2]
    dpi_dq[1,2] = -q[1]/q[2]
    dpi_dq[2,2] = 0
    dpi_dq[3,2] = -q[3]/q[2]
    return dpi_dq/q[2]

def circle_dot(q):
    c_d = np.zeros([4,6])
    c_d[0:3,0:3] = np.eye(3)
    c_d[0:3,3:6] = -up_hat(q[0:3])
    return c_d

def up_hat_pose(q):
    Q = np.zeros([4,4])
    Q[0:3,0:3] = up_hat(q[3:6])
    Q[0:3,3] = q[0:3,0]
    return Q