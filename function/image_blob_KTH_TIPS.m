function [ data_blob, lable_blob ] = image_blob_KTH_TIPS(train_stage, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
% output
    %  lable_blob
%     config.images=2;
%     config.meta=1:10;
%     dataset_root = config.dataset_root;

%     %train 
%     train_stage = 1;
%     config.train_minibatch=2;
    
%     %test
%     train_stage = 0;
%     config.id = 1;
    
    if (train_stage)
        patch_size = 224;
        % read config
        minibatch = config.train_minibatch;
        % 
        data_blob = zeros(patch_size,patch_size,3,minibatch);
        lable_blob = ones(1,1,1,minibatch);

        id = datasample(config.train_ids,1);
%         id = datasample(config.train_ids( find(config.train_ids<=1280) ),1); 
%         id = 1;
        image = single(config.IMG{1,id});
        % resize to at least 256, and keep ratio aspect (as the paper
        % roughly mentioned)
        [h,w,~] = size(image);   min_hw = min(h,w);  ratio = 256 / min_hw;
        if (ratio > 1)
            image = imresize(image,ratio);
        end
%         imshow(image);
%         disp(num2str(config.images.class(1,id)));
%         disp(config.meta.classes{1,config.images.class(1,id)});
        
        for batch = 1:minibatch
            begin1 = randi(size(image,1)-patch_size+1,1);
            begin2 = randi(size(image,2)-patch_size+1,1);
            subimage = image(begin1:begin1+patch_size-1,begin2:begin2+patch_size-1,:);
            
            lable_blob(:,:,1,batch) = config.images.class(1,id);
            data_blob(:,:,:,batch) = subimage;
        end
        data_blob = permute(data_blob, [2, 1, 3, 4]);
        data_blob = data_blob(:,:,[3, 2, 1],:);
%         lable_blob = 0;
    else
        dataset_root = config.dataset_root;
        id = config.id;
%         id = 1;
        data_blob = single(imread([dataset_root,config.images.name{1,id}]));
        
        % resize to at least 256, and keep ratio aspect (as the paper
        % roughly mentioned)
        [h,w,~] = size(data_blob);   min_hw = min(h,w);  ratio = 256 / min_hw;
        if (ratio > 1)
            data_blob = imresize(data_blob,ratio);
        end
        
        data_blob = permute(data_blob, [2, 1, 3]);
        data_blob = data_blob(:,:,[3, 2, 1],:);
        lable_blob = 0;
    end
    
    data_blob(:,:,1,:) = data_blob(:,:,1,:) - 104;
    data_blob(:,:,2,:) = data_blob(:,:,2,:) - 117;
    data_blob(:,:,3,:) = data_blob(:,:,3,:) - 123;
    data_blob = single(data_blob);
    lable_blob = uint8(lable_blob-1);

end

