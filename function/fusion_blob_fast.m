function [ haptic_blob, image_blob, lable_blob ] = fusion_blob_fast(train_stage, config)
%fusion_blob ramdomly generate data for training fusionNet

%     %train 
%     train_stage = 1;
%     config.image_cache_root = '_cache/image';
%     config.minibatch=5;
%     config.select_from=1:10;
%     load('_cache/haptic/DCTgrams.mat')
%     config.DCTgrams = DCTgrams;
%     config.max_sample_perimage = 10;
    
%     %test
%     train_stage = 0;
%     config.i = 1;
%     config.j = 1;
%     load('_cache/haptic/DCTgrams.mat')
    
    if (train_stage)
        % read config
        minibatch = config.minibatch;
        select_from = config.select_from;
        image_input_shape = [224,224]; % output size should be 1*1
        haptic_input_shape = [50,192];   % output size should be 1*1
      
        i = randi(69,1);
        j_haptic = datasample(select_from,1);
        haptic_signal = config.DCTgrams{i,j_haptic};
        j_image = datasample(select_from,1);
        sample_image = datasample(config.image_set.subimg_per_image,1);
        img_signal = config.image_set.set{i,j_image,sample_image};
        
        image_blob = zeros(image_input_shape(2),image_input_shape(1),3,minibatch);
        haptic_blob = zeros(haptic_input_shape(2),haptic_input_shape(1),1,minibatch);
        lable_blob = i*ones(1,1,1,minibatch);
        
        for batch = 1:minibatch
            % haptic signal
            begin_t = randi(size(haptic_signal,2)-haptic_input_shape(2)+1,1);
            haptic_subsignal = haptic_signal(:,begin_t:begin_t+haptic_input_shape(2)-1);
            haptic_blob(:,:,:,batch) = permute(haptic_subsignal,[2,1]);
            % image signal
            begin_w = randi(size(img_signal,2)-image_input_shape(2)+1,1);
            begin_h = randi(size(img_signal,1)-image_input_shape(1)+1,1);
            image_blob(:,:,:,batch) = img_signal(begin_h:begin_h+image_input_shape(1)-1,begin_w:begin_w+image_input_shape(2)-1,:);
        end
        
        haptic_blob = single(haptic_blob)*250-160;
        image_blob(:,:,1,:) = image_blob(:,:,1,:) - 104;
        image_blob(:,:,2,:) = image_blob(:,:,2,:) - 117;
        image_blob(:,:,3,:) = image_blob(:,:,3,:) - 123;
        image_blob = single(image_blob);
        lable_blob = uint8(lable_blob-1);
    else
        i = config.i;
        j = config.j;
        
        image_blob = single(imread([config.image_cache_root,'/','test','/',num2str(i),'/',num2str(j),'.jpg']));
        image_blob = image_blob(:,:,[3, 2, 1]);
        image_blob = permute(image_blob, [2, 1, 3]);
        image_blob(:,:,1,:) = image_blob(:,:,1,:) - 104;
        image_blob(:,:,2,:) = image_blob(:,:,2,:) - 117;
        image_blob(:,:,3,:) = image_blob(:,:,3,:) - 123;
        
        haptic_blob = config.DCTgrams{config.i,config.j};
        haptic_blob = permute(haptic_blob,[2,1]);
        haptic_blob = haptic_blob*250-160;
        haptic_blob = single(haptic_blob);
        
        lable_blob = 0;
    end
    
end

