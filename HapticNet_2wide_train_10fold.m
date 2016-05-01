% clc;
run startup;


%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = 2;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

%% init  
% load data;
load('_cache/haptic/DCTgrams.mat')

% init train data setting
train_data_config.minibatch = 10;
train_data_config.input_shape = [50,300];
train_data_config.DCTgrams=DCTgrams;

% init test data setting
test_data_config.DCTgrams=DCTgrams;

% init logging
experiment_performance{1,1} = 0;

% train 10 networks at the same time, insane thing to do..
snapshot_step = 5000;
snapshot_prefix = 'FCN';

% for generate training sequences for 10 experiments
    %  9 sets, 69 class,
rng(11111);
len = 9*69;
train_shuffle_index = datasample(1:len,len,'Replace',false);
rng('default');

train_sequence={};
for test_set = 1:10
    train_set = [1:test_set-1,test_set+1:10];
    [set, class] = ...
                 ndgrid(train_set,1:69);
             
    set = set(:);
    set = set(train_shuffle_index);
    class = class(:);
    class = class(train_shuffle_index);
             
    train_sequence{test_set}.set=set;
    train_sequence{test_set}.class=class;
    train_sequence{test_set}.len = len;
end

for count = 18:400
    for test_set = 1:10
        disp(['validation ',num2str(test_set),' time ',...
                            num2str(count)]);
        %% skip this iteration if model file have already exist
        model_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                        num2str(count),'.caffemodel'];
        snapshot_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                        num2str(count),'.solverstate'];  
        if (exist(model_name, 'file') &&  exist(snapshot_name, 'file'))
            load('models/HapticNet_2wide_10fold/log.mat');   % if so, one has to load the pre-exist log file.
            disp('  model file have already exist, skip this round');
            continue;
        end
        %% train/test setting
        train_set = [1:test_set-1,test_set+1:10];
        train_data_config.select_from=train_set;
        %% training sequence
        train_data_config.train_sequence = train_sequence{test_set};
        %% set gpu, load solver, load snapshot, reshape blob
        caffe.set_mode_gpu();
        
        solver_file = 'models/HapticNet_2wide_10fold/FCN_solver.prototxt';
        caffe_solver = caffe.Solver(solver_file);
        
        if (count>1)   %
            dst_model_name = ['models/HapticNet_2wide_10fold/FCN_iter_',num2str((count-1)*snapshot_step),...
                            '.caffemodel'];
            src_model_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.caffemodel'];
            copyfile(src_model_name, dst_model_name);
            snapshot_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count-1),'.solverstate'];
            caffe_solver.restore(snapshot_name);
            delete(dst_model_name);
        end
        
        caffe_solver.net.blobs('data').reshape([300,50,1,10]); % reshape blob 'data'
        caffe_solver.net.blobs('label').reshape([8,1,1,10]); % reshape blob 'data'
        caffe_solver.net.reshape();
        
        %% training
        for iter = 1:snapshot_step
            train_data_config.iteration = caffe_solver.iter();
            % generate training blob
            [data_blob,label_blob] = haptic_blob_2wide(true,train_data_config);
            caffe_solver.net.blobs('data').set_data(data_blob);
            caffe_solver.net.blobs('label').set_data(label_blob);

            %1 iter ADAM update
            caffe_solver.step(1);
        end
        
        %% rename model file,snapshot
        dst_model_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count),'.caffemodel'];
        dst_snapshot_name = ['models/HapticNet_2wide_10fold/model/validation',num2str(test_set),'_time',...
                            num2str(count),'.solverstate'];
        src_model_name = ['models/HapticNet_2wide_10fold/FCN_iter_',num2str(caffe_solver.iter()),...
                            '.caffemodel'];
        src_snapshot_name = ['models/HapticNet_2wide_10fold/FCN_iter_',num2str(caffe_solver.iter()),...
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
                [data_blob,label_blob] = haptic_blob_2wide(false,test_data_config);

                caffe_solver.test_nets(1).blobs('data').reshape([size(data_blob,1),size(data_blob,2),1,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).blobs('label').reshape([1,1,1,1]); % reshape blob 'data'
                caffe_solver.test_nets(1).reshape();

                caffe_solver.test_nets(1).blobs('data').set_data(data_blob);
                caffe_solver.test_nets(1).blobs('label').set_data(label_blob);

                caffe_solver.test_nets(1).forward_prefilled();
                prob = caffe_solver.test_nets(1).blobs('ip3').get_data();
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
        save('models/HapticNet_2wide_10fold/log','experiment_performance');
        disp(['     acc ',num2str(avg_acc_fragment)]);
        
        %close caffe
        caffe.reset_all();
    end
end
