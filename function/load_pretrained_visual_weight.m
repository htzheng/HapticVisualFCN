function load_pretrained_visual_weight( net, pretrained_net, suffix ) %transform_fc_layer
% load weights of each convolutional layers

if ~exist('suffix','var')
    suffix = 'VisualNet_';
end

for k = 1:numel(pretrained_net.layer_names)
    layer_name = pretrained_net.layer_names{k,1};
    layer_type = pretrained_net.layers(layer_name).type;
    if strcmp(layer_type,'Convolution')
        newlayer_name = [suffix,layer_name];
        if any(strcmp(net.layer_names,newlayer_name))
            disp([layer_name,'-->',newlayer_name]);
            net.params(newlayer_name, 1).set_data(pretrained_net.params(layer_name,1).get_data);
            net.params(newlayer_name, 2).set_data(pretrained_net.params(layer_name,2).get_data);
        else
            disp([newlayer_name,' do not dexist']);
        end
    end
end

% if ~exist('suffix','transform_fc_layer')
%             net.params(newlayer_name, 1).set_data(pretrained_net.params(layer_name,1).get_data);
%             net.params(newlayer_name, 2).set_data(pretrained_net.params(layer_name,2).get_data);
% end

end
