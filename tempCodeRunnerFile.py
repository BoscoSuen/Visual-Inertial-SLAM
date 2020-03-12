                # if((joint_mu[:, i]-w_T_imu[:,3,i]).T.dot(joint_mu[:, i]-w_T_imu[:,3,i]))>200000:
                #     joint_mu[:, i]= np.array([0,0,0,1]).T
                #     landmark_flag[i] = False