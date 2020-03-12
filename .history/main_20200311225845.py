import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *
from EKF_process import predict
from scipy.linalg import expm
from converting import up_hat,dpi_dq,get_pi,get_calibration_matrix,circle_dot,up_hat_pose

# if __name__ == '__main__':
filename = "./data/0027.npz"
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

N = features.shape[1]
mu = np.eye(4)
sigma = 0.05*np.eye(6)

joint_mu = np.zeros([4,N])	# vectorize the update
joint_sigma = 0.05*np.eye(3*N+6)

print(joint_sigma.shape)
w_T_imu = np.zeros([4,4,len(t[0])-1])
D = np.zeros([4,3])
D[0:3,0:3] = np.eye(3)
landmark_flag = np.array([False] * N)

M = get_calibration_matrix(K,b)

delta_t_list = []	# delta_t is different
for i in range(len(t[0])-1):
    delta_t_list.append(t[0][i+1] - t[0][i])

for idx in range(len(t[0]) - 1):
    # (a) IMU Loctransforms3d.euleralization via EKF Prediction
    print("time stamp idx: " + str(idx))
    # print(sum(landmark_flag))
    V_factor = 8000
    W_factor = 0.05
    W = np.eye(6) * W_factor	# motion model
    V = np.eye(4) * V_factor	# observation model
    delta_t = delta_t_list[idx]
    cur_linear_velocity = linear_velocity[:,idx]
    cur_rotation_velocity = rotational_velocity[:,idx]
    # from piazza: you should implement the covariance propagation in part(a)
    mu,joint_sigma = predict(mu,joint_sigma,delta_t,cur_linear_velocity,cur_rotation_velocity,W,N)
    # (b) Landmark Mapping via EKF Update
    # w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose
    # cur_z = features[:,:,idx]

    valid_idx = []
    for j in range(N):
        if ((features[0,j,idx] == -1 and features[1,j,idx] == -1 and features[2,j,idx] == -1 and features[3,j,idx] == -1) or (landmark_flag[j] == False)):
            continue
        else:
            valid_idx.append(j)
    if (len(valid_idx) > 0):
        z = np.zeros([4,len(valid_idx)])
        z_curve_hat = np.zeros([4,len(valid_idx)])
        H = np.zeros([len(valid_idx)*4,3*N+6])
        I_V = np.zeros([len(valid_idx)*4,len(valid_idx)*4])
        for i in range(len(valid_idx)):
            if (landmark_flag[valid_idx[i]] == True):
                q = cam_T_imu @ (mu @ joint_mu[:,valid_idx[i]])
                dq = dpi_dq(q)
                z_curve_hat[:,i] = M @ get_pi(q)
                z[:,i] = features[:,valid_idx[i],idx]		# idx is the time stamp index
                H[4*i:4*i+4,3*N:3*N+6] = M @ dq @ cam_T_imu @ circle_dot(mu @ joint_mu[:,valid_idx[i]])
                H[4*i:4*i+4,3*valid_idx[i]:3*valid_idx[i]+3] = M @ dq @ (cam_T_imu @ mu) @ D
                I_V[4*i:4*i+4,4*i:4*i+4] = V

        # Perform the EKF update
        # update K(t+1|t)
        K = joint_sigma @ np.transpose(H) @ (np.linalg.inv(H @ (joint_sigma @ np.transpose(H))+I_V))
        # update u(t+1|t)
        mu = expm(up_hat_pose(K[3*N:3*N+6,:] @ ((z-z_curve_hat).reshape(-1,1,order='F')))) @ mu
        joint_mu = joint_mu + D @ ((K[0:3*N,:] @ ((z-z_curve_hat).reshape(-1,1,order='F'))).reshape(3,-1,order='F'))
        joint_sigma = joint_sigma - (K @ H @ joint_sigma)

    w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose

    for i in range(N):
        if (not(features[0,i,idx] == -1 and features[1,i,idx] == -1 and features[2,i,idx] == -1 and features[3,i,idx] == -1)):
            if landmark_flag[i] == False:
                d = features[0, i, idx] - features[2, i, idx]
                z = K[0, 0] * b / d
                optical_bp = np.array([(features[0, i, idx] - K[0, 2]) * z / K[0, 0], (features[1, i, idx] - K[1, 2]) * z / K[1, 1], z, 1])

                joint_mu[:, i] = np.linalg.inv(mu) @ (np.linalg.inv(cam_T_imu)) @ (np.transpose(optical_bp))
                landmark_flag[i] = True
                if((joint_mu[:, i]-w_T_imu[:,3,idx]).T.dot(joint_mu[:, i]-w_T_imu[:,3,idx]))>200000:
                    joint_mu[:, i]= np.array([0,0,0,1]).T
                    landmark_flag[i] = False
print(joint_mu)
visualize_trajectory_2d(w_T_imu,joint_mu[0,:],joint_mu[1,:],path_name="Path",show_ori=True)