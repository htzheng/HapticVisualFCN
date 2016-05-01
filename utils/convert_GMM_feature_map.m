function [ iFV_features ] = convert_GMM_feature_map( feature_map,rf_size,means,covariances,priors )
% please ref http://www.vlfeat.org/api/gmm-fundamentals.html for equations
[~,d,h,w] = size(feature_map);
n = size(means,2);
feature_map = reshape(feature_map,[d,h,w]);

% GMM_feature_map = zeros(n,h,w);
% log_p_k_x = zeros(n,h,w);
% % first calculate log p(x|mean_k,cov_k) for every Gaussian kernel G_k(k=1..n)
% for k = 1:n
%     cov = covariances(:,k);
%     inv_cov = 1./cov;
%     m = means(:,k);
%     feature_map_minus_mu = feature_map - repmat(m,[1,h,w]);
%     log_p_k_x(k,:,:) = - 0.5*( d * log(2*pi) + sum(log(cov))) ....
%                         -0.5 * sum( (feature_map_minus_mu).^2 .* repmat(inv_cov,[1,h,w]) , 1 ); %.* inv_cov;
%      
% end
% 
% % then normalize to calculate log q_ik
% denominator = zeros(1,h,w);
% for k = 1:n
%     denominator = denominator + priors(k) * exp()
% end

count = 0;
fprintf('iFV ');
for i = 1 : h-rf_size+1
    fprintf('.');
    for j = 1 : w-rf_size+1
        feature = feature_map(:,i:i+rf_size-1,j:j+rf_size-1);
        feature = reshape(feature,[d,rf_size*rf_size]);
        iFV_feature = vl_fisher(feature, means, covariances, priors,'Improved');
        count = count + 1; 
        if ~exist('iFV_features','var')
            iFV_features = zeros(size(iFV_feature,1),(h-rf_size+1)*(w-rf_size+1),'single');
        end
        iFV_features(:,count) = iFV_feature;
    end
end
fprintf('\n');
% feature_map = reshape(feature_map,[d,h*w]);

% fisher_feature_map = vl_fisher(feature_map(:,1), means, covariances, priors,'Improved');
% GMM_feature_map = reshape(fisher_feature_map,[n,h,w]);
end

