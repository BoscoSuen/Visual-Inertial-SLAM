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

# (a) IMU Loctransforms3d.euleralization via EKF Prediction

# (b) Landmark Mapping via EKF Update

# (c) Visual-Inertial SLAM

# You can use the function below to visualize the robot pose over time
# visualize_trajectory_2d(world_T_imu,show_ori=True)
N = features.shape[1]
mu = np.eye(4)
sigma = 0.05*np.eye(6)	

joint_mu = np.zeros([4,N])	# vectorize the update
joint_sigma = 0.05*np.eye(3*N+6)

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
	V_factor = 0
	W_factor = 0
	W = np.eye(6) * W_factor	# motion model
	V = np.eye(4) * V_factor	# observatin model
	delta_t = delta_t_list[idx]
	cur_linear_velocity = linear_velocity[:,i]
	cur_rotation_velocity = rotational_velocity[:,i]
	# from piazza: you should implement the covariance propagation in part(a)
	mu,sigma = predict(mu,sigma,delta_t,cur_linear_velocity,cur_rotation_velocity,W,N)
	joint_sigma[3*N:3*N+6,3*N:3*N+6] = sigma

	# (b) Landmark Mapping via EKF Update
	w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose
	# cur_z = features[:,:,idx]
	
	valid_idx = []
	for i in range(N):
		if ((features[0,i,idx] == -1 and features[1,i,idx] == -1 and features[2,i,idx] == -1 and features[3,i,idx] == -1) or (landmark_flag[i] == False)):
			continue
		else:
			valid_idx.append(i)

	if (len(valid_idx) == 0): continue
	else:
		z = np.empty(shape=[4,0])
		z_curve_hat = np.empty(shape=[4,0])
		H = np.zeros([len(valid_idx)*4,3*N+6])
		I_V = np.zeros([len(valid_idx)*4,len(valid_idx)*4])
		for i in range(len(valid_idx)):
			if (landmark_flag[i] is True):
				q = np.dot(cam_T_imu,np.dot(mu,joint_mu[:,valid_idx[i]]))
				dq = dpi_dq(q)
				z_curve_hat = np.append(z_curve_hat,np.dot(M,get_pi(q)),axis=1)
				z = np.append(z,features[:,valid_idx[i],idx])		# idx is the time stamp index
				# TODO: need update the z with z_curve_hat?
				H[4*i:4*i+4,3*N:3*N+6] = np.dot(M,np.dot(dq,np.dot(cam_T_imu,circle_dot(np.dot(mu,joint_mu[:,valid_idx[i]])))))
				H[4*i:4*i+4,3*valid_idx[i]:3*valid_idx[i]+3] = np.dot(M,np.dot(dq,np.dot(np.dot(cam_T_imu,mu),D)))
				I_V[4*i:4*i+4,4*i:4*i+4] = V

		# Perform the EKF update
		# update K(t+1|t)
        K = joint_sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(joint_sigma.dot(np.transpose(H)))+I_V))
		# update u(t+1|t)
        mu = scipy.linalg.expm(up_hat_pose(K[3*N:3*N+6,:].dot((z-z_curve_hat).reshape(-1,1,order='F')))).dot(mu)
        joint_mu = joint_mu + D.dot(K[0:3*M,:].dot((z-z_curve_hat).reshape(-1,1,order='F'))).reshape(3,-1,order='F')
        joint_sigma = joint_sigma - (K.dot(H).dot(joint_sigma))

	w_T_imu[:,:,idx] = np.linalg.inv(mu)	# Ut is the inverse IMU pose

	