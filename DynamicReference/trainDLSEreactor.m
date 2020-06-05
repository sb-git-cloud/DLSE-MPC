clear all
close all
clc

yr = 3e4;
load(['dataset',num2str(yr),'.mat']);

numNeur = 30;

u = trainInp;
y = Jdata;

trf = 'trainbr';

net = feedforwardnet(numNeur,trf);
tpam = net.trainParam;
tpam.max_fail = 1e2;
tpam.epochs = 1e4;

minperf = Inf;

for ii = 1:10
    [net_temp, Temp_temp, netPar_temp, fun_temp] = ...
        trainDLSE(u,y,numNeur,trf,tpam);
    perf = perform(net,u,y);
    if perf < minperf
        minperf = perf;
        net = net_temp;
        netPar = netPar_temp;
    end    
end

figure()
plot((y - net(u))./y)

save(['dlseReactor',num2str(yr),'.mat'],'netPar','net')