import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *
from converting import up_hat
from scipy.linalg import expm

# if __name__ == '__main__':
filename = "./data/0027.npz"
t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

# (a) IMU Loctransforms3d.euleralization via EKF Prediction

# (b) Landmark Mapping via EKF Update

# (c) Visual-Inertial SLAM

# You can use the function below to visualize the robot pose over time
# visualize_trajectory_2d(world_T_imu,show_ori=True)
mu = np.eye(4)
N = features.shape[1]		# 3950
cov = 0.05 * np.eye(3*N+6)	# match the shape of H to calculate the K gain

w_T_imu = np.zeros([4,4,len(t[0])-1])

delta_t_list = []	# delta_t is different
for i in range(len(t[0])-1):
	delta_t_list.append(t[0][i+1] - t[0][i])

for idx in range(len(t[0]) - 1):
	# (a) IMU Loctransforms3d.euleralization via EKF Prediction
	V_factor = 3000
	W_factor = 0.1
	V = np.eye(4) * V_factor
	W = np.eye(6) * V_factor
	delta_t = delta_t_list[idx]
	cur_linear_velocity = linear_velocity[:,i]
	cur_rotation_velocity = rotational_velocity[:,i]
	ut_up_hat = np.zeros([4,4])		# pose kinematics
	ut_up_hat[0:3,0:3] = up_hat(cur_rotation_velocity)
	ut_up_hat[0:3,3] = cur_linear_velocity
	mu = expm(-delta_t * ut_up_hat).dot(mu)
	w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose

	# from piazza: you should implement the covariance propagation in part(a)
	ut_curly_up_hat = np.zeros([6,6])
	ut_curly_up_hat[0:3,0:3] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[3:6,3:6] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[0:3,3:6] = up_hat(cur_linear_velocity)
	temp = np.dot(cov[3*N:3*N+6,3*N:3*N+6],np.transpose(expm(-delta_t*ut_curly_up_hat)))
	cov[3*N:3*N+6,3*N:3*N+6] = np.dot(expm(-delta_t * ut_curly_up_hat),temp)

	# (b) Landmark Mapping via EKF Update