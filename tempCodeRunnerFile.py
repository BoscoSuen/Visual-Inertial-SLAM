    for i in range(N):
        if (not(features[0,i,idx] == -1 and features[1,i,idx] == -1 and features[2,i,idx] == -1 and features[3,i,idx] == -1)):
            if landmark_flag[i] == False:
                d = features[0, i, idx] - features[2, i, idx]
                z = K[0, 0] * b / d
                optical_bp = np.array([(features[0, i, idx] - K[0, 2]) * z / K[0, 0], (features[1, i, idx] - K[1, 2]) * z / K[1, 1], z, 1])

                land_mark[:, i] = np.linalg.inv(mu) @ (np.linalg.inv(cam_T_imu)) @ (np.transpose(optical_bp))
                landmark_flag[i] = True