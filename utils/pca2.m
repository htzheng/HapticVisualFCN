function [ output_args ] = pca2( data )

center_data = mean(data,2);
data = bsxfun(@minus, data, center_data);

end

