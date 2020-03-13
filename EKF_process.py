import numpy as np
from converting import up_hat
from scipy.linalg import expm
from converting import dpi_dq,get_pi,circle_dot

def predict(mu,joint_sigma,delta_t,cur_linear_velocity,cur_rotation_velocity,W,N):
	ut_up_hat = np.zeros([4, 4])  # pose kinematics
	ut_up_hat[0:3, 0:3] = up_hat(cur_rotation_velocity)
	ut_up_hat[0:3, 3] = cur_linear_velocity
	mu = expm(-delta_t * ut_up_hat) @ mu

	ut_curly_up_hat = np.zeros([6, 6])
	ut_curly_up_hat[0:3, 0:3] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[3:6, 3:6] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[0:3, 3:6] = up_hat(cur_linear_velocity)
	joint_sigma[3*N:3*N+6,3*N:3*N+6] = expm(-delta_t*ut_curly_up_hat) @ (joint_sigma[3*N:3*N+6,3*N:3*N+6] @ (expm(-delta_t*ut_curly_up_hat).T)) + delta_t**2 * W

	return mu,joint_sigma

def update(z,z_curve_hat,H,I_V,cam_T_imu,joint_mu,M,features,mu,valid_idx,D,V,N,idx):
	for i in range(len(valid_idx)):
		q = cam_T_imu @ (mu @ joint_mu[:, valid_idx[i]])
		dq = dpi_dq(q)
		z_curve_hat[:, i] = M @ get_pi(q)
		z[:, i] = features[:, valid_idx[i], idx]  # idx is the time stamp index
		# avoid singular matrix
		if ((z_curve_hat[:, i] - z[:, i]).T.dot(z_curve_hat[:, i] - z[:, i])) > 10000:
			z[:, i] = z_curve_hat[:, i]
		H[4 * i:4 * i + 4, 3 * N:3 * N + 6] = M @ dq @ cam_T_imu @ circle_dot(mu @ joint_mu[:, valid_idx[i]])
		H[4 * i:4 * i + 4, 3 * valid_idx[i]:3 * valid_idx[i] + 3] = M @ dq @ (cam_T_imu @ mu) @ D
		I_V[4 * i:4 * i + 4, 4 * i:4 * i + 4] = V

	return z,z_curve_hat,H,I_V