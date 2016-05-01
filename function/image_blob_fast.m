function [ data_blob, lable_blob ] = image_blob_fast(train_stage, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
% output
    %  lable_blob

%     %train 
%     train_stage = 1;
%     config.minibatch=2;
%     config.select_from=1:10;
%     config.
%     config.cache_root = '_cache/image';
%     config.max_sample_perimage = 10;
    
%     %test
%     train_stage = 0;
%     config.i = 1;
%     config.j = 1;
%     config.cache_root = '_cache/image';

    if (train_stage)
        % read config
        minibatch = config.minibatch;
        select_from = config.select_from;
        
        %calculate output shape (height, width)
	    input_shape = config.input_shape;
        output_shape = config.output_shape;
        
        data_blob = zeros(input_shape,input_shape,3,minibatch);
        lable_blob = zeros(minibatch,1,output_shape,output_shape);
        
        
        for batch = 1:minibatch
            i = randi(69,1);
            j = datasample(select_from,1);
            iter = randi(config.image_set.subimg_per_image,1);
            image = single(config.image_set.set{i,j,iter});
            image = image(:,:,[3, 2, 1]);
            image = permute(image, [2, 1, 3]);
            
            begin1 = randi(size(image,1)-input_shape+1,1);
            begin2 = randi(size(image,2)-input_shape+1,1);
            
            data_blob(:,:,:,batch) = image(begin1:begin1+input_shape-1,begin2:begin2+input_shape-1,:);
            lable_blob(batch,1,:,:) = i;
        end
        lable_blob = permute(lable_blob,[4,3,2,1]);
    else

    end
    data_blob(:,:,1,:) = data_blob(:,:,1,:) - 104;
    data_blob(:,:,2,:) = data_blob(:,:,2,:) - 117;
    data_blob(:,:,3,:) = data_blob(:,:,3,:) - 123;
    data_blob = single(data_blob);
    lable_blob = uint8(lable_blob-1);

end

