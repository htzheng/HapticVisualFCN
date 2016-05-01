function [ PCA_featuremap ] = PCA_reduce_featuremap(feature_map,coeff,explained,mu,PcaDimensions)
    [~,d,h,w] = size(feature_map);
    feature_map = reshape(feature_map,[d,h*w])';
    
    coeff = coeff(:,1:PcaDimensions);
%     PCA_featuremap = (feature_map - repmat(mu,[h*w,1])) * coeff;
    PCA_featuremap = bsxfun(@minus, feature_map, mu) * coeff;
    
    PCA_featuremap = PCA_featuremap';
    
    PCA_featuremap = reshape(PCA_featuremap,[1,PcaDimensions,h,w]);
end