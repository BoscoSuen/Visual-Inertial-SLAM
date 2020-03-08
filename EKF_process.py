import numpy as np
from converting import up_hat
from scipy.linalg import expm

def predict(mu,sigma,delta_t,cur_linear_velocity,cur_rotation_velocity,W,N):
	ut_up_hat = np.zeros([4, 4])  # pose kinematics
	ut_up_hat[0:3, 0:3] = up_hat(cur_rotation_velocity)
	ut_up_hat[0:3, 3] = cur_linear_velocity
	mu = expm(-delta_t * ut_up_hat).dot(mu)

	ut_curly_up_hat = np.zeros([6, 6])
	ut_curly_up_hat[0:3, 0:3] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[3:6, 3:6] = up_hat(cur_rotation_velocity)
	ut_curly_up_hat[0:3, 3:6] = up_hat(cur_linear_velocity)
	temp = np.dot(sigma[0:6,0:6], np.transpose(expm(-delta_t * ut_curly_up_hat)))
	sigma[0:6,0:6] = np.dot(expm(-delta_t * ut_curly_up_hat), temp) + W

	return mu,sigma