% clc;
run startup;


%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

%% init  
% load data;
load('_cache/haptic/DCTgrams.mat')

% init train data setting
train_data_config.minibatch = 1;
train_data_config.image_cache_root = '_cache/image';
train_data_config.max_sample_perimage = 180;
train_data_config.DCTgrams = DCTgrams;

% init test data setting
test_data_config.image_cache_root = '_cache/image';
test_data_config.DCTgrams = DCTgrams;

% init logging
experiment_performance{1,1} = 0;

% train config
snapshot_step = 5000;    %%%%%%%%%%%%%%%%%%%
snapshot_prefix = 'FCN';

% fusion sampling number 
fusion_sample_number = 1000;

for count = 1:20
    %% prepare images
    disp('preparing image set...');
    image_prepare_config.cache_root = '_cache/image';
    [ image_set] = image_blob_fast_prepare(snapshot_step, train_data_config.minibatch, train_data_config.max_sample_perimage, image_prepare_config);
    train_data_config.image_set = image_set;
    
    for test_set = 10
        %% train/test setting
        train_set = [1:test_set-1,test_set+1:10];
        train_data_config.select_from=train_set;

        %% set gpu, load solver, load snapshot(or pretrained model), reshape blob
        caffe.set_mode_gpu();
        
        solver_file = 'models/FusionNet_TCNN_10fold/FusionNet_69dim_solver.prototxt';
        caffe_solver = caffe.Solver(solver_file);
        
        if (count>1)   %
            dst_model_name = ['models/FusionNet_TCNN_10fold/FCN_69dim_iter_',num2str((count-1)*snapshot_step),...
                            '.caffemodel'];
            src_model_name = ['models/FusionNet_TCNN_10fold/model_69dim/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.caffemodel'];
            copyfile(src_model_name, dst_model_name);
            snapshot_name = ['models/FusionNet_TCNN_10fold/model_69dim/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.solverstate'];
            caffe_solver.restore(snapshot_name);
            delete(dst_model_name);
        else    % load pretrained weight from HapticNet and VisualNet
            Haptic_count = 20;
            Haptic_model = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(Haptic_count),'.caffemodel'];
            caffe_solver.net.copy_from(Haptic_model);
            
            Visual_count = 20;
            Visual_model = ['models/TCNN_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(Visual_count),'.caffemodel'];
            caffe_solver.net.copy_from(Visual_model);
        end
        
        caffe_solver.net.blobs('data').reshape([192,50,1,train_data_config.minibatch]); % reshape blob 'data'
        caffe_solver.net.blobs('VisualNet_data').reshape([224,224,3,train_data_config.minibatch]); % reshape blob 'data'
        caffe_solver.net.blobs('label').reshape([1,1,1,train_data_config.minibatch]); % reshape blob 'data'
        caffe_solver.net.reshape();
        
        %% training
        for iter = 1:snapshot_step
            % generate training blob
%             tic;
%             [ haptic_blob, image_blob, lable_blob ] = fusion_blob(true,train_data_config);
%             toc;

%             tic;
            [haptic_blob, image_blob, lable_blob] = fusion_blob_fast(true,train_data_config);
%             toc;

            caffe_solver.net.blobs('data').set_data(haptic_blob);
            caffe_solver.net.blobs('VisualNet_data').set_data(image_blob);
            caffe_solver.net.blobs('label').set_data(lable_blob);
%             tic;
            %1 iter ADAM update
            caffe_solver.step(1);
%             toc;
        end
        
        %% rename model file,snapshot
        dst_model_name = ['models/FusionNet_TCNN_10fold/model_69dim/validation',num2str(test_set),'_time',...
                            num2str(count),'.caffemodel'];
        dst_snapshot_name = ['models/FusionNet_TCNN_10fold/model_69dim/validation',num2str(test_set),'_time',...
                            num2str(count),'.solverstate'];
        src_model_name = ['models/FusionNet_TCNN_10fold/FCN_69dim_iter_',num2str(caffe_solver.iter()),...
                            '.caffemodel'];
        src_snapshot_name = ['models/FusionNet_TCNN_10fold/FCN_69dim_iter_',num2str(caffe_solver.iter()),...
                            '.solverstate'];           
        movefile(src_model_name, dst_model_name);
        movefile(src_snapshot_name, dst_snapshot_name);
        
        % testing and save performance
        caffe_solver.test_nets(1).copy_from(dst_model_name);
        fusion_layer_net = caffe.Net('models/FusionNet_TCNN_10fold/fusion_69dim_layer.prototxt', dst_model_name, 'test'); % create net and load weights
        
        % do valdiation per val_interval iterations
        acc_fragment = zeros(69,1);
        acc_track = zeros(69,1);
        cofusion_matrix_fragment = zeros(69,69);
        cofusion_matrix_track = zeros(69,69);
        for i = 1:69
            for j = test_set
                test_data_config.i = i;
                test_data_config.j = j;
                [haptic_blob, image_blob, lable_blob] = fusion_blob_fast(false,test_data_config);

                caffe_solver.test_nets(1).blobs('data').reshape([size(haptic_blob,1),size(haptic_blob,2),1,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).blobs('VisualNet_data').reshape([size(image_blob,1),size(image_blob,2),3,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).reshape();

                caffe_solver.test_nets(1).blobs('data').set_data(haptic_blob);
                caffe_solver.test_nets(1).blobs('VisualNet_data').set_data(image_blob);

                % get image haptic feature maps
                caffe_solver.test_nets(1).forward_prefilled();
                haptic_feature_map = caffe_solver.test_nets(1).blobs('ip3').get_data();
                visual_feature_map = caffe_solver.test_nets(1).blobs('TCNN_fc8').get_data();
                
                % sample features
                haptic_feature_map_random = haptic_feature_map(...
                            randi(size(haptic_feature_map,1),[fusion_sample_number,1]),:,:);
                visual_feature_map = reshape(visual_feature_map,...
                            [size(visual_feature_map,1)*size(visual_feature_map,2),1,size(visual_feature_map,3)]);
                visual_feature_map_random = visual_feature_map(...
                            randi(size(visual_feature_map,1),[fusion_sample_number,1]),:,:);
                
                % forward fusion layer
                fusion_layer_net.blobs('ip3').set_data(haptic_feature_map_random);
                fusion_layer_net.blobs('TCNN_fc8').set_data(visual_feature_map_random);
                fusion_layer_net.forward_prefilled();
                
                prob = fusion_layer_net.blobs('FusionNet_fc').get_data();
                [~,label] = max(prob,[],3);
                label=label(:);

                %accuracy
                acc_fragment(i,1) = sum(label==i)/size(label,1);
                max_vote = mode(label);
                acc_track(i,1) = max_vote==i;

                % cofusion-matrix
                for k=1:69
                    cofusion_matrix_fragment(i,k)=cofusion_matrix_fragment(i,k)+sum(label==k);
                end
                % cofusion-matrix-track
                cofusion_matrix_track(i,mode(label))=cofusion_matrix_track(i,mode(label))+1;
            end
        end
        avg_acc_fragment = mean(acc_fragment);
        avg_acc_track = mean(acc_track);

        log.cofusion_matrix_fragment = cofusion_matrix_fragment;
        log.cofusion_matrix_track = cofusion_matrix_track;
        log.avg_acc_fragment = avg_acc_fragment;
        log.avg_acc_track = avg_acc_track;
        log.iteration = caffe_solver.iter();
        experiment_performance{test_set,count} = log; %record the performance(validation,time_stamp)
        save('models/FusionNet_TCNN_10fold/log_69dim','experiment_performance');
        disp(['validation ',num2str(test_set),' time ',...
                            num2str(count),'     acc ',num2str(avg_acc_fragment)]);
        %close caffe
        caffe.reset_all();
    end
end
