function varargout = trainDLSE(u,y,hiddenSize,trainFcn,trainPar)
%TRAINDLSE train a DLSE neural network
%
% [net, Temp, netPar, fun] = TRAINDLSE(u,y,hiddenSize,trainFcn,trainPar) takes 
% u: MxN input vector,
% y: N output vector, 
% hiddenSize: number of nodes in the hidden layer,
% trainFcn: training function, 
% trainPar: training parameters, 
% and returns 
% net: a trained DLSET neural network  
% Temp: the temperature parameter of the network
% netPar: the parameters of the DLSET network
% fun: the convex and the concave function
%
% Defaults are used if TRAINDLSE is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'

switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        hiddenSize = 10;
        net = dlsenet(u, y);
    case 3
        net = dlsenet(u, y, hiddenSize);
    case 4
        net = dlsenet(u, y, hiddenSize, trainFcn);
    case 5
        net = dlsenet(u, y, hiddenSize, trainFcn);
        net.trainParam = trainPar;
end

[net,~,~,~] = train(net, u, y);

switch nargout
    case 1 
        varargout{1} = net;
        
    case 2
        varargout{1} = net;
        Temp = 1/net.outputs{2}.processSettings{1}.gain;
        varargout{2} = Temp;
        
    case 3
        varargout{1} = net;
        Temp = 1/net.outputs{3}.processSettings{1}.gain;
        varargout{2} = Temp;
        Alpha = net.IW{1,1};
        Beta = net.b{1};

        uoff = net.inputs{1}.processSettings{1}.xoffset;
        ugain = net.inputs{1}.processSettings{1}.gain;
        umin = net.inputs{1}.processSettings{1}.ymin;

        yoff = net.outputs{3}.processSettings{1}.xoffset;
        ygain = net.outputs{3}.processSettings{1}.gain;
        ymin = net.outputs{3}.processSettings{1}.ymin;
      
        Alpha1 = Alpha(1:hiddenSize,:);
        Alpha2 = Alpha(1+hiddenSize:end,:);

        Beta1 = Beta(1:hiddenSize,:);
        Beta2 = Beta(1+hiddenSize:end,:);
        
        netPar = struct('Alpha1',Alpha1,'Alpha2',Alpha2,'Beta1',Beta1,...
            'Beta2',Beta2,'uoff',uoff,'ugain',ugain,'umin',umin,...
            'yoff',yoff,'ygain',ygain,'ymin',ymin);
        varargout{3} = netPar;
    
    case 4
        varargout{1} = net;
        Temp = 1/net.outputs{3}.processSettings{1}.gain;
        varargout{2} = Temp;
        Alpha = net.IW{1,1};
        Beta = net.b{1};

        uoff = net.inputs{1}.processSettings{1}.xoffset;
        ugain = net.inputs{1}.processSettings{1}.gain;
        umin = net.inputs{1}.processSettings{1}.ymin;

        yoff = net.outputs{3}.processSettings{1}.xoffset;
        ygain = net.outputs{3}.processSettings{1}.gain;
        ymin = net.outputs{3}.processSettings{1}.ymin;
      
        Alpha1 = Alpha(1:hiddenSize,:);
        Alpha2 = Alpha(1+hiddenSize:end,:);

        Beta1 = Beta(1:hiddenSize,:);
        Beta2 = Beta(1+hiddenSize:end,:);
        
        Dugain = diag(ugain);
        
        netPar = struct('Alpha1',Alpha1,'Alpha2',Alpha2,'Beta1',Beta1,...
            'Beta2',Beta2,'uoff',uoff,'ugain',ugain,'umin',umin,...
            'yoff',yoff,'ygain',ygain,'ymin',ymin);
        varargout{3} = netPar;
        
        convfun = @(x) ( log(sum(exp(Alpha1*(Dugain*(x - uoff) + umin) + Beta1),1)) - ymin/2)/ygain + yoff/2;
        concfun = @(x) (-log(sum(exp(Alpha2*(Dugain*(x - uoff) + umin) + Beta2),1)) - ymin/2)/ygain + yoff/2;
        
        functs = struct('convfun',convfun,'concfun',concfun);
        varargout{4} = functs;
end

end