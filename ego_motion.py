import numpy as np
import matplotlib.pyplot as plt
import cv2 

def Motion_detect(Image1, Image2, Height, K):
    '''
    Image1: image 1 
    Image2: image 2
    Height: Height of ego car in metres
    K: 3x3 camera intrinsic matrix

    '''

    orb = cv2.ORB_create()
    Keypoints1, Descriptors1 = orb.detectAndCompute(Image1, None)
    Keypoints2, Descriptors2 = orb.detectAndCompute(Image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(Descriptors1,Descriptors2)
    
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)
    kp1_list = np.mat([])
    kp2_list = np.mat([])
    k = 0

    number_of_matches = 3000

    for m in matches:
        img1Idx = m.queryIdx
        img2Idx = m.trainIdx

        (img1x, img1y) = Keypoints1[img1Idx].pt
        (img2x, img2y) = Keypoints2[img2Idx].pt

        if k == 0:
            kp1_list = [[img1x,img1y,1]]
            kp2_list = [[img2x,img2y,1]]
            k = 1
        else:
            kp1_list = np.append(kp1_list,[[img1x,img1y,1]],axis = 0)
            kp2_list = np.append(kp2_list,[[img2x,img2y,1]],axis = 0)
            k+=1
        if k == number_of_matches:
            break

    F = cv2.findFundamentalMat(kp1_list[:,0:2], kp2_list[:,0:2], method = cv2.FM_RANSAC)
    E = K.T @ F[0] @ K 
    
    Image3 = cv2.drawMatches(Image1,kp1_list,Image2,kp2_list,matches[:10]ls
    )
    plt.imshow(Image3)
    plt.show()
    Transformation_info = cv2.recoverPose(E, (kp1_list[:,0:2]), (kp2_list[:,0:2]), K)
    Rotation = Transformation_info[1]
    Translation = Transformation_info[2]
    Translation = Translation# / Translation[2]

    return Rotation, Translation

if __name__ == "__main__":

    File1 = '/home/rahul/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/0000/000000.png'
    File2 = '/home/rahul/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/0000/000001.png'

    Image1 = plt.imread(File1)
    Image2 = plt.imread(File2)
    plt.imshow(Image1)
    plt.show()

    H = 1.65
    K  = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02],
    [0.000000e+00, 7.215377e+02, 1.728540e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    R, T = Motion_detect(Image1, Image2, H, K )

    print(R,'\n',T)


