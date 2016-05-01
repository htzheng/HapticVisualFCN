function [ data_blob, lable_blob ] = haptic_blob_2wide(train_stage, config)
%haptic_blob ramdomly generate data for training hapticNet
% input
    % cache_root
    % minibatch - the number of minibatch
    % select_from - a list, from 1 to 10 where the sample are drawn from
% output
    %  lable_blob

%     %train 
%     train_stage = 1;
%     config.minibatch=10;
%     config.select_from=1:10;
%     config.input_shape = [50,300];
%     load('/media/haitian/WD Elements/TUM_texture_new/_cache/haptic/DCTgrams.mat')
    
%     %test
%     train_stage = 0;
%     config.i = 1;
%     config.j = 1;
%     load('/media/haitian/WD Elements/TUM_texture_new/_cache/haptic/DCTgrams.mat')
    
    if (train_stage)
        % read config
        minibatch = config.minibatch;
        select_from = config.select_from;
        input_shape = config.input_shape;
        iter = config.iteration;
        
        %calculate output shape (height, width)
        output_shape = ceil(input_shape/2);
        output_shape = ceil(output_shape/2);
        output_shape = ceil(output_shape/2);
        output_shape = ceil(output_shape/2);
        output_shape = [output_shape(1)-4+1, output_shape(2)-12+1];
        
        data_blob = zeros(minibatch,1,input_shape(1),input_shape(2));
        lable_blob = zeros(minibatch,1,output_shape(1),output_shape(2));
        
        for batch = 1:minibatch
%             i = randi(69,1);  %%%%%%%%%%%%%%%%%69
%             j = datasample(select_from,1);
            
            i = config.train_sequence.class(mod(iter,config.train_sequence.len)+1);
            j = config.train_sequence.set(mod(iter,config.train_sequence.len)+1);
%             scatter(config.train_sequence.class,config.train_sequence.set);
%             pause();
            if (isempty(find(j==select_from)))
               asdfasdfasdf();
            end
            
            shape = size(config.DCTgrams{i,j});
            length = shape(2);
            begin_t = randi(length-input_shape(2)+1,1);
            
            signal = config.DCTgrams{i,j};
            signal = signal(:,begin_t:begin_t+input_shape(2)-1);
            
            data_blob(batch,1,:,:) = signal;
            lable_blob(batch,1,:,:) = i;
        end
        data_blob = permute(data_blob,[4,3,2,1]);
        lable_blob = permute(lable_blob,[4,3,2,1]);
    else
        signal = config.DCTgrams{config.i, config.j};
        input_shape = size(signal);
        data_blob = zeros(1,1,input_shape(1),input_shape(2));
        data_blob(1,1,:,:) = signal;
        data_blob = permute(data_blob,[4,3,2,1]);
        lable_blob = config.i;
    end
    data_blob = single(data_blob)*250-160;
    lable_blob = uint8(lable_blob-1);
end

