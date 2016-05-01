function [ PCA_features ] = PCA_reduce_features(features,projection,mean_x)
    PCA_features = projection * bsxfun(@minus, features, mean_x) ;
end