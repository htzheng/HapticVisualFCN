function [ data_blob, lable_blob ] = image_blob_DTD(train_stage, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
% output
    %  lable_blob
%     config.images=2;
%     config.meta=1:10;

%     %train 
%     train_stage = 1;
%     config.train_minibatch=2;
    
%     %test
%     train_stage = 0;
%     config.id = 1;

    if (train_stage)
        % read config
        minibatch = config.train_minibatch;
        dataset_root = config.dataset_root;
        
        % 
        data_blob = zeros(224,224,3,minibatch);
        lable_blob = ones(1,1,1,minibatch);
        
        
        
%         lable_blob = lable_blob * config.images.class(1,id);
        
        for batch = 1:minibatch
            id = datasample(config.train_ids,1);
            image = single(imread([dataset_root,'images/',config.images.name{1,id}]));
            image = image(:,:,[3, 2, 1]);
            image = permute(image, [2, 1, 3]);
            lable_blob(1,1,1,batch) = config.images.class(1,id);
            
            begin1 = randi(size(image,1)-224+1,1);
            begin2 = randi(size(image,2)-224+1,1);
            data_blob(:,:,:,batch) = image(begin1:begin1+224-1,begin2:begin2+224-1,:);
        end
    else
        dataset_root = config.dataset_root;
        id = config.id;
        image = single(imread([dataset_root,'images/',config.images.name{1,id}]));
        image = image(:,:,[3, 2, 1]);
        
        data_blob = permute(image, [2, 1, 3]);
        lable_blob = 0;
    end
    data_blob(:,:,1,:) = data_blob(:,:,1,:) - 104;
    data_blob(:,:,2,:) = data_blob(:,:,2,:) - 117;
    data_blob(:,:,3,:) = data_blob(:,:,3,:) - 123;
    data_blob = single(data_blob);
    lable_blob = uint8(lable_blob-1);

end

