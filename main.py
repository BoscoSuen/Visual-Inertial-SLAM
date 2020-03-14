import numpy as np
import cv2,os
from tqdm import tqdm
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *
from EKF_process import predict,update
from scipy.linalg import expm
from converting import up_hat,dpi_dq,get_pi,get_calibration_matrix,circle_dot,up_hat_pose

# if __name__ == '__main__':
filename = "./data/0027.npz"
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

N = features.shape[1]
mu = np.eye(4)
sigma = np.zeros([6,6])

land_mark = np.zeros([4,N])	# landmark
joint_sigma = 0.05*np.eye(3*N+6)
joint_sigma[3*N:3*N+6,3*N:3*N+6] = sigma

# print(joint_sigma.shape)
w_T_imu_pred = np.zeros([4,4,len(t[0])-1])
w_T_imu_update = np.zeros([4,4,len(t[0])-1])
D = np.zeros([4,3])
D[0:3,0:3] = np.eye(3)
landmark_flag = np.array([False] * N)

M = get_calibration_matrix(K,b)

delta_t_list = []	# delta_t is different
for i in range(len(t[0])-1):
    delta_t_list.append(t[0][i+1] - t[0][i])

for idx in tqdm(range(len(t[0]) - 1)):
    # (a) IMU Loctransforms3d.euleralization via EKF Prediction
    # print("time stamp idx: " + str(idx))
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
    w_T_imu_pred[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose
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

        # Perform the EKF update
        z,z_curve_hat,H,I_V = update(z,z_curve_hat,H,I_V,cam_T_imu,land_mark,M,features,mu,valid_idx,D,V,N,idx)

        # update K(t+1|t)
        K_t = joint_sigma @ np.transpose(H) @ (np.linalg.inv((H @ (joint_sigma @ np.transpose(H)))+I_V))
        # update u(t+1|t)
        mu = expm(up_hat_pose(K_t[3*N:3*N+6,:] @ ((z-z_curve_hat).reshape(-1,1,order='F')))) @ mu
        land_mark = land_mark + D @ ((K_t[0:3*N,:] @ ((z-z_curve_hat).reshape(-1,1,order='F'))).reshape(3,-1,order='F'))
        joint_sigma = (np.eye(K_t.shape[0]) - K_t @ H) @ joint_sigma

    w_T_imu_update[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose

    for i in range(N):
        if (not(features[0,i,idx] == -1 and features[1,i,idx] == -1 and features[2,i,idx] == -1 and features[3,i,idx] == -1)):
            if landmark_flag[i] == False:
                d = features[0, i, idx] - features[2, i, idx]
                z = K[0, 0] * b / d
                optical_bp = np.array([(features[0, i, idx] - K[0, 2]) * z / K[0, 0], (features[1, i, idx] - K[1, 2]) * z / K[1, 1], z, 1])

                land_mark[:, i] = np.linalg.inv(mu) @ (np.linalg.inv(cam_T_imu)) @ (np.transpose(optical_bp))
                landmark_flag[i] = True
                # avoid singular matrix
                # if((land_mark[:, i]-w_T_imu_update[:,3,idx]).T.dot(land_mark[:, i]-w_T_imu_update[:,3,idx]))>10000:
                #     land_mark[:, i]= np.array([0,0,0,1]).T
                #     landmark_flag[i] = False

# (c) Visual-Inertial SLAM
visualize_trajectory_2d_pose(w_T_imu_pred,path_name="Path",show_ori=True,path_save="output/pose_pred.png")
visualize_trajectory_2d_pose(w_T_imu_update,path_name="Path",show_ori=True,path_save="output/pose_update.png")
visualize_trajectory_2d(w_T_imu_pred,land_mark[0,:],land_mark[1,:],path_name="Path",show_ori=True,path_save="output/landmark_pred.png")
visualize_trajectory_2d(w_T_imu_update,land_mark[0,:],land_mark[1,:],path_name="Path",show_ori=True,path_save="output/landmark_upate.png")