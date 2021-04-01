function outnet = dlsenet(u,y,hiddenSize,trainFcn)
%DLSENET design a neural network as a differnce of LSE functions
%
% DLSENET(u,y,hiddenSize,trainFcn)  takes 
% u: MxN input vector,
% y: N output vector, 
% hiddenSize: number of nodes in the hidden layer,
% trainFcn: training function, 
% and returns a DLSET feed-forward neural network.
%
% Defaults are used if DLSENET is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'


switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        hiddenSize = 10;
        net = feedforwardnet([2*hiddenSize,2]);
    case 3
        net = feedforwardnet([2*hiddenSize,2]);
    case 4
        net = feedforwardnet([2*hiddenSize,2],trainFcn);
end

net = configure(net,'inputs',u);
net = configure(net,'outputs',y);

net.biasConnect = [1; 0; 0];

conWeig = zeros(size(net.LW{2,1}));
conWeig(1,1:hiddenSize) = ones(1,hiddenSize);
conWeig(2,1+hiddenSize:end) = ones(1,hiddenSize);

net.LW{2,1} = conWeig;

net.layerWeights{2,1}.learn = 0;

net.LW{3,2} = [1, -1]; 

net.layerWeights{3,2}.learn = 0;

net.layers{1}.transferFcn = 'expfun';

net.layers{2}.transferFcn = 'logfun';

outnet = net;

end