function [ data_blob, lable_blob ] = image_blob_Kylberg(train_stage, config)
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
        % read config
        minibatch = config.train_minibatch;
        
        % 
        data_blob = zeros(512,512,minibatch);
        lable_blob = ones(10,10,1,minibatch);

        id = datasample(config.train_ids,1);
%         id = datasample(config.train_ids( find(config.train_ids<=1280) ),1); 
%         id = 1;
        image = single(config.IMG{1,id});
        
%         imshow(image);
%         disp(num2str(config.images.class(1,id)));
%         disp(config.meta.classes{1,config.images.class(1,id)});
        
        for batch = 1:minibatch
            begin1 = randi(size(image,1)-512+1,1);
            begin2 = randi(size(image,2)-512+1,1);
            subimage = image(begin1:begin1+512-1,begin2:begin2+512-1);
            
            lable_blob(:,:,1,batch) = config.images.class(1,id);
            data_blob(:,:,batch) = subimage;
        end
        data_blob = reshape(data_blob,[512,512,1,minibatch]);
        data_blob = repmat(data_blob,[1,1,3,1]);
        data_blob = permute(data_blob, [2, 1, 3, 4]);
%         lable_blob = 0;
    else
        dataset_root = config.dataset_root;
        id = config.id;
%         id = 1;
        data_blob = single(imread([dataset_root,config.images.name{1,id}]));
        data_blob = repmat(data_blob,[1,1,3]);
        data_blob = permute(data_blob, [2, 1, 3]);

        lable_blob = 0;
    end
    
    data_blob(:,:,1,:) = data_blob(:,:,1,:) - 104;
    data_blob(:,:,2,:) = data_blob(:,:,2,:) - 117;
    data_blob(:,:,3,:) = data_blob(:,:,3,:) - 123;
    data_blob = single(data_blob);
    lable_blob = uint8(lable_blob-1);

end

