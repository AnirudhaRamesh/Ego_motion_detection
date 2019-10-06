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
    Inputs:
        X1 - 3 x N - 3D world points corresponding to image 1
        X2 - 3 x N - 3D world points corresponding to image 2
        R - 3 x 3 - Rotation of ego car pose 2 with respect to pose 1
        T - 3 x 1 - Translation of ego car pose 2 with respect to pose 1

    Output:
        scale - 1xN - scaling factor for translation vector
    '''
    # Minimizing square distance and choosing scale factor 
    scale = ((X1 - R @ X2).T @ T) / (T.T @ T)
        
    return scale

def Scale_trajectory(K, HeightOfCam, Start_frame, End_frame, threshold, sequence_number):


    '''
    K - 3x3 - intrinsic matrix
    HeighOfCam - 1x1 height of camera
    Start_frame - initial frame
    End_frame - Ending frame
    threshold - depth threshold of points
    sequence_number - number of sequence

    This function plots scaled trajectory of the ORB SLam output vs ground truth.
    '''
    invK = np.linalg.inv(K)

    file_out = './scales/%d_scales.txt' % sequence_number
    f = open(file_out,'wb')
    ORBSLAM_file = './ORBSLAM_output/KITTITrajectoryComplete_new_%d' % sequence_number
    DataPose = np.loadtxt(ORBSLAM_file)
    Ground_truth = './Ground_truth/%d.txt' % sequence_number
    actualdata = np.loadtxt(Ground_truth,delimiter=',')
    # print(actualdata.shape)
    
    
    
    seqDir = './lane_tracks/%04d' % sequence_number
    
    #ORB pose cummulative matrix initialization without scaling
    C_orb = np.eye(4)
    
    # THRESHOLD
    Threshold = threshold

    #Our scaled ORB pose cummulative matrix initialization
    Ccumu = np.eye(4)

    #Ground truth pose cummulative matrix initialization
    Ca_cummu = np.eye(4) 
    Trans = np.array([0, 0, 0])
    TransActual = np.array([0, 0, 0])
    T_orb = np.array([0,0,0])

    Rgt_init = np.reshape(actualdata[Start_frame, :9],(3,3))
    Tgt_init = np.array([actualdata[Start_frame, 9:]]).T     
    Init_transformation = np.vstack((np.hstack((Rgt_init.T, -Rgt_init @ Tgt_init)),np.array([0, 0, 0, 1])))
    for i in range(End_frame - Start_frame - 1):

        # npy file contains corresponding points in the image
        points_file_path = seqDir + '/%06d_tracks.npy' % (Start_frame + i)
        # print(points_file_path)

        DataPixels = np.load(points_file_path)

        # Taking corresponding points of current frame and next frame
        Pixels_1 = np.hstack((DataPixels[0,:,:],np.ones((DataPixels[0,:,:].shape[0],1))))
        Pixels_2 = np.hstack((DataPixels[1,:,:],np.ones((DataPixels[1,:,:].shape[0],1))))
        
        # print(Pixels_1.shape)
        
        Pose1 = DataPose[Start_frame + i,:]
        Pose2 = DataPose[Start_frame + i + 1,:]


        # Estimating pose using Chandrakar's formula
        X1 = PointEstimate(invK, Pixels_1.T, Height)

        # Applying threshold X1 and X2 are 3xN matrices of world points in camera frame 1 and 2 respectively (corresponding to the same pixels).
        tempArray = (X1[2,:] >= Threshold) 
        X1 = X1[:,tempArray]
        X2 = PointEstimate(invK, Pixels_2.T, Height)
        X2 = X2[:,tempArray]

        # Getting ORB pose
        R1 = np.mat([[Pose1[1],Pose1[2],Pose1[3]],[Pose1[4],Pose1[5],Pose1[6]],[Pose1[7],Pose1[8],Pose1[9]]]) 
        T1 = np.mat([[Pose1[10]],[Pose1[11]],[Pose1[12]]])
        
        C1 = np.vstack((np.hstack((R1, T1)),np.array([0, 0, 0, 1])))
        

        R2 = np.mat([[Pose2[1],Pose2[2],Pose2[3]],[Pose2[4],Pose2[5],Pose2[6]],[Pose2[7],Pose2[8],Pose2[9]]]) 
        T2 = np.mat([[Pose2[10]],[Pose2[11]],[Pose2[12]]])

        C2 = np.vstack((np.hstack((R2, T2)),np.array([0, 0, 0, 1])))
        
        invC = np.vstack((np.hstack((R1.T, -R1.T @ T1)),np.array([0, 0, 0, 1])))
        C_change = invC @ C2
        C_orb = C_orb @ C_change
        
        T_orb = np.vstack((T_orb,C_orb[:3,3].T))

        
        R = C_change[:3,:3]
        T = C_change[:3,3]

        T = T / np.linalg.norm(T)

        scale = ScaleEstimate(X1, X2, R, T)
        # print(scale.shape)
        np.savetxt(f, scale.T, delimiter=', ', newline='')
        

        Rgt_current = np.reshape(actualdata[Start_frame + i + 1, :9],(3,3))
        Tgt_current = np.array([actualdata[Start_frame + i + 1, 9:]]).T 
        
        # previous frame's rotation and translation
        Rgt_prev = np.reshape(actualdata[Start_frame + i, :9],(3,3))
        Tgt_prev = np.array([actualdata[Start_frame + i, 9:]]).T

        #current pose
        actualPose_current = np.vstack((np.hstack((Rgt_current.T, Tgt_current)),np.array([0, 0, 0, 1])))
        # print(actualPose_current)

        # inverse of ground truth of previous pose
        invCa_1 = np.vstack((np.hstack((Rgt_prev, -Rgt_prev @ Tgt_prev)),np.array([0, 0, 0, 1])))

        # getting only current transformation matrix
        Ca_ = invCa_1 @ actualPose_current

        # Cummulative pose matrix
        Ca_cummu = Ca_cummu @ Ca_
        
        # average scaling factor        
        avgscale = np.mean(scale)
        
        Tcorr = avgscale * T
        C = np.vstack((np.hstack((R, Tcorr)),np.array([0, 0, 0, 1])))
        Ccumu = Ccumu @ C
        
        Trans = np.vstack((Trans,Ccumu[:3,3].T))
        TransActual = np.vstack((TransActual,Ca_cummu[:3,3].T))
        
    # print(T_orb)
    # print(Trans)
    # print(TransActual)
    plt.plot(Trans[:,0],Trans[:,2],'b-o',label = 'scaled')
    plt.plot(TransActual[:,0],TransActual[:,2],'r-o',label = 'Ground Truth')
    # DataPose[:,9:11] = DataPose[:,9:11] - DataPose[Start_frame,9:11]
    plt.plot(T_orb[:,0],T_orb[:,2],'y-o',label = 'Orb_output')

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
    
    Start_frame = 50
    End_frame = 200
    Threshold = 15
    seqid = 4
    # Sequence 5 improper trajectory
    # Sequence 4 works decently well frame 50 onwards as more cooresponding points in images
    Scale_trajectory(K, Height, Start_frame, End_frame, Threshold, seqid)
    