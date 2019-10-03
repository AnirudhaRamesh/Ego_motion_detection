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




if __name__ == "__main__":
    K  = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02],
    [0.000000e+00, 7.215377e+02, 1.728540e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    invK = np.linalg.inv(K)

    Height = 1.65
    DataPixels = np.load('000001_tracks.npy')
    DataPose = np.loadtxt('KITTITrajectoryComplete_new_3.txt')
    actualdata = np.loadtxt('3.txt',delimiter=',')
    print(actualdata.shape)
    plt.plot(actualdata[41:50,9],actualdata[41:50,11],'r',label='Ground truth')

    print(DataPose.shape)

    DataPose = DataPose[41:50,:]

    Pixels_1 = DataPixels[0,:,:]
    Pixels_2 = DataPixels[1,:,:]
    Pixels_1 = np.hstack((DataPixels[0,:,:],np.ones((Pixels_1.shape[0],1))))
    Pixels_2 = np.hstack((DataPixels[1,:,:],np.ones((Pixels_2.shape[0],1))))
    print(Pixels_1.shape)

    Trans = np.array([[]])
    # THRESHOLD
    Threshold = 20
    actual = np.array([-0.063183,-0.83695,46.425])

    for i in range(8):
        
        Pose1 = DataPose[i,:]
        Pose2 = DataPose[i + 1,:]

        X1 = PointEstimate(invK, Pixels_1.T, Height)
        tempArray = (X1[2,:] >= Threshold) 
        X1 = X1[:,(X1[2,:] >= Threshold)]
        X2 = PointEstimate(invK, Pixels_2.T, Height)
        # print(X2.shape)

        X2 = X2[:,tempArray]
        print(X2.shape)

        R1 = np.mat([[Pose1[1],Pose1[2],Pose1[3]],[Pose1[4],Pose1[5],Pose1[6]],[Pose1[7],Pose1[8],Pose1[9]]]) 
        T1 = np.mat([[Pose1[10]],[Pose1[11]],[Pose1[12]]])

        R2 = np.mat([[Pose2[1],Pose2[2],Pose2[3]],[Pose2[4],Pose2[5],Pose2[6]],[Pose2[7],Pose2[8],Pose2[9]]]) 
        T2 = np.mat([[Pose2[10]],[Pose2[11]],[Pose2[12]]])

        R = R1.T @ R2
        T = T2 - T1
        T = T / np.linalg.norm(T)

        scale = ScaleEstimate(X1, X2, R, T)
        # print(scale.shape)
        # print(scale)
        
        # sc = np.median(scale)
        # print(sc)
        # print(sc[0,int(len(scale) / 2)])
        # print(np.median(scale))

        #average scaling factor        
        avgscale = np.mean(scale)
        if i == 0:
           Trans = avgscale * T.T
        else:
           Trans = np.vstack((Trans,avgscale * T.T))

        print(avgscale * T.T)
        # print(R.shape)
        # print(T.shape)


    for i in range(1,Trans.shape[0]):

        Trans[i] = Trans[i] + Trans[i - 1]

    Trans = Trans + actual
    print(Trans)
    plt.plot(Trans[:,0],Trans[:,2],label = 'scaled')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.show()    