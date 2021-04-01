function varargout = trainRPOS(u,y,hiddenSize,trainFcn,trainPar)
%TRAINRPOS train a RPOS neural network
%
% [net, Temp, netPar, gpos,fun] = TRAINRPOS(u,y,hiddenSize,trainFcn,trainPar) takes 
% u: MxN input vector,
% y: N output vector, 
% hiddenSize: number of nodes in the hidden layer,
% trainFcn: training function, 
% trainPar: training parameters, 
% and returns 
% net: a trained RPOS neural network  
% Temp: the temperature parameter of the network
% netPar: the parameters of the network
% gpos: the generalized posynomial
% fun: the functions generating the RPOS
%
% Defaults are used if TRAINRPOS is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'

switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        hiddenSize = 10;
        net = dlsenet(ut, yt);
    case 3
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = dlsenet(ut, yt, hiddenSize);
    case 4
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = dlsenet(ut, yt, hiddenSize,trainFcn);
    case 5
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = dlsenet(ut, yt, hiddenSize,trainFcn);
        net.trainParam = trainPar;
end

[net,~,~,~] = train(net, ut, yt);

switch nargout
    case 1 
        varargout{1} = net;
    case 2
        varargout{1} = net;
        Temp = 1/net.outputs{3}.processSettings{1}.gain;
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
        
        netPar = struct('Alpha1',Alpha1,'Alpha2',Alpha2,'Beta1',Beta1,...
            'Beta2',Beta2,'uoff',uoff,'ugain',ugain,'umin',umin,...
            'yoff',yoff,'ygain',ygain,'ymin',ymin);
        varargout{3} = netPar;
        
        varargout{4} = @(x) exp(net(log(x)));
        
    case 5
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
        
        varargout{4} = @(x) exp(net(log(x)));
        
        Dugain = diag(ugain);
        
        num = @(x) exp(yoff - ymin/ygain)*(sum(exp(Alpha1*(Dugain*(log(x) - uoff) + umin) + Beta1),1)).^(1/ygain);
        den = @(x) (sum(exp(Alpha2*(Dugain*(log(x) - uoff) + umin) + Beta2),1)).^(1/ygain);
        
        functs = struct('num',num,'den',den);
        varargout{5} = functs;
end



end