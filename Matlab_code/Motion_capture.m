function [R, T] = Motion_capture(Image1, Image2, Height, K)

    points1 = detectSURFFeatures(Image1,'MetricThreshold');
    points2 = detectSURFFeatures(Image2,'MetricThreshold');
    
    
    [f1,vpts1] = extractFeatures(Image1,points1);
    [f2,vpts2] = extractFeatures(Image2,points2);
    
    indexPairs = matchFeatures(f1,f2);
    matchedPoints1 = vpts1(indexPairs(:,1),:);
    matchedPoints2 = vpts2(indexPairs(:,2),:);
    
%     figure; showMatchedFeatures(Image1,Image2,matchedPoints1,matchedPoints2);
%     legend('matched points 1','matched points 2');
%     
    cameraParams = cameraParameters('IntrinsicMatrix',K); 

    F = estimateFundamentalMatrix(matchedPoints1, matchedPoints2, 'Method','RANSAC','DistanceThreshold',1e-4);
    E = K' * F * K;
    
    [R, T] = relativeCameraPose(E, cameraParams, matchedPoints1, matchedPoints2);
    R = R';
    T = -T;
    
    Dimensions = size(matchedPoints1.Location);
    inverseK = inv(K);
    m = 1;
    alphamean = 0;
    for i = 1:Dimensions(1)
        if i == 1
            min = sqrt(([matchedPoints2.Location(i,:) 1] * F * [matchedPoints1.Location(i,:) 1]')^2);
            index = 1;
        end
        if matchedPoints1.Location(i,2) >= 200
            if sqrt(([matchedPoints2.Location(i,:) 1] * F * [matchedPoints1.Location(i,:) 1]')^2) < 0.5
                index = i;
                min = sqrt(([matchedPoints2.Location(i,:) 1] * F * [matchedPoints1.Location(i,:) 1]')^2);
                WorldPoint2 = PointEstimate(inverseK,matchedPoints1.Location(index,:),Height);
                if WorldPoint2(3)<= 15
                    correctIndex = index;
                end
            end
        end
    end
    
    WorldPoint2 = PointEstimate(inverseK,matchedPoints1.Location(correctIndex,:),Height);
    WorldPoint1 = PointEstimate(inverseK,matchedPoints2.Location(correctIndex,:),Height);
    WorldPointSkew = WorldPoint2 - WorldPoint1;
    
    RotatedWorldPoint2 = R * WorldPoint2;
    T = T / norm(T);
%     T = sqrt(WorldPointSkew(1)^2 + WorldPointSkew(3)^2) * T
    %T * (WorldPoint1 - RotatedWorldPoint2)

    alpha = T * (WorldPoint1 - RotatedWorldPoint2) / (T * T');
%     for i= 1:10
%         L = ((WorldPoint1 - (RotatedWorldPoint2 + alpha * T))^2);
%         grad = 
    
    T = 2 * alpha * T;
    Error = (WorldPoint1 - R*WorldPoint2 - T').^2;
    
end
    
    