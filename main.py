import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *
from EKF_process import predict

# if __name__ == '__main__':
filename = "./data/0027.npz"
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

# (a) IMU Loctransforms3d.euleralization via EKF Prediction

# (b) Landmark Mapping via EKF Update

# (c) Visual-Inertial SLAM

# You can use the function below to visualize the robot pose over time
# visualize_trajectory_2d(world_T_imu,show_ori=True)
N = features.shape[1]
mu = np.eye(4)
sigma = 0.05*np.eye(3*N+6)	# vectorize the update

w_T_imu = np.zeros([4,4,len(t[0])-1])
D = np.zeros([4,3])	
D[0:3,0:3] = np.eye(3)
landmark_flag = np.array([False] * N)

delta_t_list = []	# delta_t is different
for i in range(len(t[0])-1):
	delta_t_list.append(t[0][i+1] - t[0][i])

for idx in range(len(t[0]) - 1):
	# (a) IMU Loctransforms3d.euleralization via EKF Prediction
	V_factor = 0
	W_factor = 0
	W = np.eye(6) * W_factor	# motion model
	V = np.eye(4) * V_factor	# observatin model
	delta_t = delta_t_list[idx]
	cur_linear_velocity = linear_velocity[:,i]
	cur_rotation_velocity = rotational_velocity[:,i]
	# from piazza: you should implement the covariance propagation in part(a)
	mu,sigma = predict(mu,sigma,delta_t,cur_linear_velocity,cur_rotation_velocity,W,N)

	# (b) Landmark Mapping via EKF Update
	w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose
	cur_z = features[:,:,idx]
	
	valid_idx = []
	for i in range(N):
		if ((features[0,i,idx] == -1 and features[1,i,idx] == -1 and features[2,i,idx] == -1 and features[3,i,idx] == -1) or (landmark_flag[i] == False)):
			continue
		else:
			valid_idx.append(i)

	if (len(valid_idx) == 0): continue
	else:
		