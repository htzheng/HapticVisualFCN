function [ PCA_features ] = PCA_reduce_features_2(features,projection,mean_x)
    PCA_features = projection * bsxfun(@minus, features, mean_x) ;
%     for i = 1:100
%         plot(PCA_features(:,i));
%         pause();
%     end
end