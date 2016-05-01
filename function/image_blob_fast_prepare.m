function [ image_set] = image_blob_fast_prepare(training_iteration, batch_size, max_size_per_image, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
% output
% config.cache_root = '_cache/image';
% training_iteration = 5000;
% batch_size = 5;
% max_size_per_image = 10;


image_set.subimg_per_image = ceil(training_iteration*batch_size/69/9);
image_set.subimg_per_image = min(image_set.subimg_per_image, max_size_per_image);
image_set.select_list = datasample(max_size_per_image,image_set.subimg_per_image);
image_set.set = cell(69,10,image_set.subimg_per_image);

% tic;
for i = 1:69
    disp(i);
    for j = 1:10
        for k = 1:image_set.subimg_per_image
            img = imread(...
                [config.cache_root,'/','train','/',num2str(i),'/',num2str(j),'_',num2str(image_set.select_list(k)),'.jpg']);
            img = img(:,:,[3, 2, 1]);
            img = permute(img, [2, 1, 3]);
            image_set.set{i,j,k} = img;
        end
    end
end
% toc;
end

