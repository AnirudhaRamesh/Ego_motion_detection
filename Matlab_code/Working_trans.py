import numpy as np
import matplotlib.pyplot as plt


def PointEstimate(invK, x, Height):
    '''
    Inputs:
        invK - 3x3 - inverse of Camera intrinsic matrix
        x - 3xN - 2D homodeneous points
        Height - 1x1 camera height

    Output:
        3xN - 3D world points    
    '''

    # diectional vector
    dirVector = invK @ x
    # print(dirVector)

    # returns 3D world points 
    return (-Height * dirVector) / (np.array([[0,-1,0]]) @ dirVector) 

def ScaleEstimate(X1, X2, R, T):
    '''
    X1 - 3 x N - 3D world points corresponding to image 1
    X2 - 3 x N - 3D world points corresponding to image 2
    R - 3 x 3 - Rotation of ego car pose 2 with respect to pose 1
    T - 3 x 1 - Translation of ego car pose 2 with respect to pose 1

    '''
    # Minimizing square distance and choosing scale factor 
    scale = ((X1 - R @ X2).T @ T) / (T.T @ T)
        
    return scale

def Scale_trajectory(K, HeightOfCam, Corresponding_points_file, ORBSLAM_file, Ground_truth, Start_frame, End_frame, threshold):


    '''
    K - 3x3 - intrinsic matrix
    HeighOfCam - 1x1 height of camera
    Corresponding_points_file - npy file with corresponding points
    ORBSLAM_file - ORBSLAM file 
    Ground_truth - Ground_truth.txt file
    Start_frame - initial frame
    End_frame - Ending frame
    threshold - depth threshold of points

    This function plots scaled trajectory of the ORB SLam output vs ground truth.
    '''
    invK = np.linalg.inv(K)

    f = open('3_scales.txt','wb')
    DataPixels = np.load(Corresponding_points_file)
    print(DataPixels.shape)
    DataPose = np.loadtxt(ORBSLAM_file)
    actualdata = np.loadtxt(Ground_truth,delimiter=',')
    print(actualdata.shape)
    
    DataPose = DataPose[Start_frame:End_frame,:]

    Pixels_1 = DataPixels[0,:,:]
    Pixels_2 = DataPixels[1,:,:]
    Pixels_1 = np.hstack((DataPixels[0,:,:],np.ones((Pixels_1.shape[0],1))))
    Pixels_2 = np.hstack((DataPixels[1,:,:],np.ones((Pixels_2.shape[0],1))))
    print(Pixels_1.shape)

    Trans = np.array([[]])
    # THRESHOLD
    Threshold = threshold
    # actual = np.array([-0.063183,-0.83695,46.425])
    Ccumu = np.eye(4)
    Ca_cummu = np.eye(4) 
    Trans = np.array([0, 0, 0])
    TransActual = np.array([0, 0, 0])
    for i in range(End_frame - Start_frame - 1):
        
        Pose1 = DataPose[i,:]
        Pose2 = DataPose[i + 1,:]

        X1 = PointEstimate(invK, Pixels_1.T, Height)
        tempArray = (X1[2,:] >= Threshold) 
        X1 = X1[:,(X1[2,:] >= Threshold)]
        X2 = PointEstimate(invK, Pixels_2.T, Height)
        # print(X2.shape)

        X2 = X2[:,tempArray]
        R1 = np.mat([[Pose1[1],Pose1[2],Pose1[3]],[Pose1[4],Pose1[5],Pose1[6]],[Pose1[7],Pose1[8],Pose1[9]]]) 
        T1 = np.mat([[Pose1[10]],[Pose1[11]],[Pose1[12]]])
        C1 = np.vstack((np.hstack((R1, T1)),np.array([0, 0, 0, 1])))
        

        R2 = np.mat([[Pose2[1],Pose2[2],Pose2[3]],[Pose2[4],Pose2[5],Pose2[6]],[Pose2[7],Pose2[8],Pose2[9]]]) 
        T2 = np.mat([[Pose2[10]],[Pose2[11]],[Pose2[12]]])

        C2 = np.vstack((np.hstack((R2, T2)),np.array([0, 0, 0, 1])))
        invC = np.vstack((np.hstack((R1.T, -R1.T @ T1)),np.array([0, 0, 0, 1])))
        C_change = invC @ C2
        R = C_change[:3,:3]
        T = C_change[:3,3]

        T = T / np.linalg.norm(T)

        scale = ScaleEstimate(X1, X2, R, T)
        print(scale.shape)
        np.savetxt(f, scale.T, delimiter=', ', newline='')
        # f.write('\n')

        Ractual = actualdata[Start_frame + i, :9]
        Ractual = np.reshape(Ractual,(3,3))
        Tactual = np.array([actualdata[Start_frame + i, 9:]]).T

        Ractual2 = actualdata[Start_frame + i - 1, :9]
        Ractual2 = np.reshape(Ractual,(3,3))
        Tactual2 = np.array([actualdata[Start_frame + i - 1, 9:]]).T    
        actualPose_1 = np.vstack((np.hstack((Ractual2, Tactual2)),np.array([0, 0, 0, 1])))
        actualPose = np.vstack((np.hstack((Ractual, Tactual)),np.array([0, 0, 0, 1])))
        invCa_1 = np.vstack((np.hstack((Ractual2.T, -Ractual2.T @ Tactual2)),np.array([0, 0, 0, 1])))

        Ca_ = invCa_1 @ actualPose
        Ca_cummu = Ca_cummu @ Ca_
        #average scaling factor        
        avgscale = np.mean(scale)
        
        # print(avgscale * T)
        Tcorr = avgscale * T
        C = np.vstack((np.hstack((R, Tcorr)),np.array([0, 0, 0, 1])))
        Ccumu = Ccumu @ C
        
        Trans = np.vstack((Trans,Ccumu[:3,3].T))
        TransActual = np.vstack((TransActual,Ca_cummu[:3,3].T))
        

    print(Trans)
    print(TransActual)
    plt.plot(Trans[:,0],Trans[:,2],'b-o',label = 'scaled')
    plt.plot(TransActual[:,0],TransActual[:,2],'r-o',label = 'Ground Truth')

    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.show()    
    f.close()

if __name__ == "__main__":
    K  = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02],
    [0.000000e+00, 7.215377e+02, 1.728540e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    invK = np.linalg.inv(K)

    Height = 1.65
    points_file = '000001_tracks.npy'
    ORBSLAM_file = 'KITTITrajectoryComplete_new_3.txt'
    Ground_truth_file = '3.txt'
    Start_frame = 5
    End_frame = 114
    Threshold = 15
    Scale_trajectory(K, Height,points_file, ORBSLAM_file, Ground_truth_file, Start_frame, End_frame, Threshold)
    