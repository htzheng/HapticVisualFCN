function [ data_blob, lable_blob ] = image_blob(train_stage, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
% output
    %  lable_blob

%     %train 
%     train_stage = 1;
%     config.minibatch=2;
%     config.select_from=1:10;
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
        max_sample_perimage = config.max_sample_perimage;
        cache_root = config.cache_root;
        iter = config.iteration;
        %calculate output shape (height, width)
	    input_shape = config.input_shape;
        output_shape = config.output_shape;

        data_blob = zeros(minibatch,input_shape,input_shape,3);
        lable_blob = zeros(minibatch,1,output_shape,output_shape);
        
        for batch = 1:minibatch
%             i = randi(69,1);
%             j = datasample(select_from,1);
%             pic_num = randi(max_sample_perimage,1);
            i = config.train_sequence.class(mod(iter,config.train_sequence.len)+1);
            j = config.train_sequence.set(mod(iter,config.train_sequence.len)+1);
            pic_num = config.train_sequence.image_idx(mod(iter,config.train_sequence.len)+1);
%             scatter3(config.train_sequence.class,config.train_sequence.set,config.train_sequence.image_idx);
%             pause();
            
            if (isempty(find(j==select_from)))
               asdfasdfasdf();
            end
            
            image = single(imread([cache_root,'/','train','/',num2str(i),'/',num2str(j),'_',num2str(pic_num),'.jpg']));
            
            begin1 = randi(size(image,1)-input_shape+1,1);
            begin2 = randi(size(image,2)-input_shape+1,1);
            
            data_blob(batch,:,:,:) = image(begin1:begin1+input_shape-1,begin2:begin2+input_shape-1,:);
            lable_blob(batch,1,:,:) = i;
            
%             imshow(squeeze(data_blob(batch,:,:,:))/255);
%             pause();
        end
        data_blob = permute(data_blob, [1,4,2,3]);
        data_blob = data_blob(:,[3, 2, 1],:,:);
        data_blob = permute(data_blob, [4,3,2,1]);
        lable_blob = permute(lable_blob,[4,3,2,1]);
    else
        cache_root = config.cache_root;
        i = config.i;
        j = config.j;
        
        signal = single(imread([cache_root,'/','test','/',num2str(i),'/',num2str(j),'.jpg']));
        [height,width,~] = size(signal);
        signal = signal(:,:,[3, 2, 1]);
        signal = permute(signal, [3, 1, 2]);
                
        data_blob = zeros(1,3,height,width);
        data_blob(1,:,:,:) = signal;
        data_blob = permute(data_blob,[4,3,2,1]);
        lable_blob = i;
    end
    data_blob(:,:,1,:) = data_blob(:,:,1,:) - 104;
    data_blob(:,:,2,:) = data_blob(:,:,2,:) - 117;
    data_blob(:,:,3,:) = data_blob(:,:,3,:) - 123;
    data_blob = single(data_blob);
    lable_blob = uint8(lable_blob-1);

end

