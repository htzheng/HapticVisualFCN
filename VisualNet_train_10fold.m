% clc;
run startup;


%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = 1 %auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

%% init  
% init train data setting
train_data_config.minibatch = 2;
train_data_config.cache_root = '_cache/image';
train_data_config.max_sample_perimage = 180;  %%  !!!!!!!!!!!!!!! should be 180
train_data_config.input_shape = 384;
train_data_config.output_shape = 6;

% init test data setting
test_data_config.cache_root = '_cache/image';

% init logging
experiment_performance{1,1} = 0;

% train 10 networks at the same time, insane thing to do..
snapshot_step = 5000;                         %% !!!!!!!!!!!!!!!! should be 5000
snapshot_prefix = 'FCN';

% for generate training sequences for 10 experiments
    %  9 sets, 69 class, max_sample_perimage images
rng(11111);
len = 9*69*train_data_config.max_sample_perimage;
train_shuffle_index = datasample(1:len,len,'Replace',false);
rng('default');

train_sequence={};
for test_set = 1:10
    train_set = [1:test_set-1,test_set+1:10];
    [set, class, image_idx] = ...
                 ndgrid(train_set,1:69,1:train_data_config.max_sample_perimage);
             
    set = set(:);
    set = set(train_shuffle_index);
    class = class(:);
    class = class(train_shuffle_index);
    image_idx = image_idx(:);
    image_idx = image_idx(train_shuffle_index);
             
    train_sequence{test_set}.set=set;
    train_sequence{test_set}.class=class;
    train_sequence{test_set}.image_idx=image_idx;
    train_sequence{test_set}.len = len;
end


for count = 1:100
    %% prepare images
    disp('preparing image set...');
%     image_prepare_config.cache_root = '_cache/image';
%     [ image_set] = image_blob_fast_prepare(snapshot_step, train_data_config.minibatch, train_data_config.max_sample_perimage, image_prepare_config);
%     train_data_config.image_set = image_set;

%     %% regenerate train list every epoch
%     % prepare train image list, 10 set, 69 class, max_sample_perimage images
%     [list_set,list_class,list_image_idx] = ...
%                 meshgrid(1:10,1:69,1:train_data_config.max_sample_perimage);

    
    for test_set = 1:10
        disp(['validation ',num2str(test_set),' time ',...
                            num2str(count)]);
        %% skip this iteration if model file have already exist
        model_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                        num2str(count),'.caffemodel'];
        snapshot_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                        num2str(count),'.solverstate'];  
        if (exist(model_name, 'file') &&  exist(snapshot_name, 'file'))
            load('models/VisualNet_10fold/log.mat');   % if so, one has to load the pre-exist log file.
            disp('  model file have already exist, skip this round');
            continue;
        end
        %% train/test setting
        train_set = [1:test_set-1,test_set+1:10];
        train_data_config.select_from=train_set;

        %% training sequence
        train_data_config.train_sequence = train_sequence{test_set};
        %% set gpu, load solver, load snapshot(or pretrained model), reshape blob
        caffe.set_mode_gpu();
        
        solver_file = 'models/VisualNet_10fold/VisualNet_solver.prototxt';
        caffe_solver = caffe.Solver(solver_file);
        
        if (count>1)   %
            dst_model_name = ['models/VisualNet_10fold/FCN_iter_',num2str((count-1)*snapshot_step),...
                            '.caffemodel'];
            src_model_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.caffemodel'];
            copyfile(src_model_name, dst_model_name);
            snapshot_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.solverstate'];
            caffe_solver.restore(snapshot_name);
            delete(dst_model_name);
        else      % load pretrained model
            pretrained_model = 'models/VisualNet_10fold/bvlc_reference_caffenet_deploy.prototxt';
            pretrained_file = 'models/VisualNet_10fold/bvlc_reference_caffenet.caffemodel';
%             pretrained_model = 'models/VisualNet_10fold/FCN_deploy.prototxt';
%             pretrained_file = 'models/VisualNet_10fold/FCN_iter_1260000.caffemodel';
            pretrained_net = caffe.Net(pretrained_model, pretrained_file, 'test'); % create net and load weights
            load_pretrained_visual_weight(caffe_solver.net, pretrained_net);
        end
        
        caffe_solver.net.blobs('VisualNet_data').reshape([train_data_config.input_shape,train_data_config.input_shape,3,train_data_config.minibatch]); % reshape blob 'data'
        caffe_solver.net.blobs('VisualNet_label').reshape([train_data_config.output_shape,train_data_config.output_shape,1,train_data_config.minibatch]); % reshape blob 'data'
        caffe_solver.net.reshape();
        
        %% training
        for iter = 1:snapshot_step
            % generate training blob
             %tic;
             train_data_config.iteration = caffe_solver.iter();
             [data_blob,label_blob] = image_blob(true,train_data_config);
             %toc;

             %tic;
%              [data_blob,label_blob] = image_blob_fast(true,train_data_config);
             %toc;
            
            caffe_solver.net.blobs('VisualNet_data').set_data(data_blob);
            caffe_solver.net.blobs('VisualNet_label').set_data(label_blob);

            %tic;
            %1 iter ADAM update
            caffe_solver.step(1);
            %toc;
        end
        
        %% rename model file,snapshot
        dst_model_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count),'.caffemodel'];
        dst_snapshot_name = ['models/VisualNet_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count),'.solverstate'];
        src_model_name = ['models/VisualNet_10fold/FCN_iter_',num2str(caffe_solver.iter()),...
                            '.caffemodel'];
        src_snapshot_name = ['models/VisualNet_10fold/FCN_iter_',num2str(caffe_solver.iter()),...
                            '.solverstate'];           
        movefile(src_model_name, dst_model_name);
        movefile(src_snapshot_name, dst_snapshot_name);
        
        % testing and save performance
        caffe_solver.test_nets(1).copy_from(dst_model_name);

        % do valdiation per val_interval iterations
        acc_fragment = zeros(69,1);
        acc_track = zeros(69,1);
        cofusion_matrix_fragment = zeros(69,69);
        cofusion_matrix_track = zeros(69,69);
        for i = 1:69
            for j = test_set
                test_data_config.i = i;
                test_data_config.j = j;
                [data_blob,label_blob] = image_blob(false,test_data_config);

                caffe_solver.test_nets(1).blobs('VisualNet_data').reshape([size(data_blob,1),size(data_blob,2),3,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).blobs('VisualNet_label').reshape([1,1,1,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).reshape();

                caffe_solver.test_nets(1).blobs('VisualNet_data').set_data(data_blob);
                caffe_solver.test_nets(1).blobs('VisualNet_label').set_data(label_blob);

                caffe_solver.test_nets(1).forward_prefilled();
                prob = caffe_solver.test_nets(1).blobs('VisualNet_fc8').get_data();
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
        save('models/VisualNet_10fold/log','experiment_performance');
        disp(['validation ',num2str(test_set),' time ',...
                            num2str(count),'     acc ',num2str(avg_acc_fragment)]);
        %close caffe
        caffe.reset_all();
    end
end
