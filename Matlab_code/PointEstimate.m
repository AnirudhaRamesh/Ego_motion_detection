function WorldPoint = PointEstimate(invK, point, Height)
    point = [point 1];
    dirVector = invK * point';
    WorldPoint = (- Height * dirVector / ([0,-1,0] * dirVector));
end