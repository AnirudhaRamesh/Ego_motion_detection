Tall = [];
for i = 1:30
    
    File1 = fullfile('~/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/',sprintf('%04d/%06d.png',0,i));
    File2 = fullfile('~/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/',sprintf('%04d/%06d.png',0,i + 1));
    
%     File1 = '~/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/0000/000000.png';
%     File2 = '~/Robotics Research Centre/KITTI dataset/data_tracking_image_2/training/image_02/0000/000001.png';

    i
    Image1 = rgb2gray(imread(File1));
    Image2 = rgb2gray(imread(File2));

    Height = 1.65;
    K = [721.53,0,609.55;0,721.53,172.85;0,0,1];

    [R,T] = Motion_capture(Image1, Image2, Height, K);
    if i ~= 1
        Tall = [Tall; Tall(i-1,:) - T];
    else
        Tall = [Tall; T];
    end
end
plot(Tall(:,1),Tall(:,3))