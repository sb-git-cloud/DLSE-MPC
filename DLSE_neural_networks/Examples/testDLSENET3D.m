clear all
close all
clc

m = 100;
noi = 0; %0.5;

u1 = linspace(-2,2);
u2 = linspace(-2,2);

dgenfun = @(u1,u2) u1.^2 + sin(2*pi*u1) - cos(2*u2);

[uu1,uu2] = meshgrid(u1,u2);
yy = dgenfun(uu1,uu2);

net = feedforwardnet(10,'trainlm');
tpam = net.trainParam;
tpam.max_fail = 1e2;

u = [uu1(:)';uu2(:)'];
y = yy(:)';

[net, Temp, netPar, fun] = trainDLSE(u,y,25,'trainlm',tpam);

%%
options = optimoptions('fmincon','OptimalityTolerance', 1e-6,...
    'Display','none','SpecifyObjectiveGradient',true);

[xstar,ystar] = dlsea(netPar,zeros(2,1),[],[],[],[],...
    -2*ones(2,1),2*ones(2,1),[],options);

[imin,jmin] = find(yy == min(min(yy)),1,'first');

fprintf('Exact minimum\n');
fprintf('argmin = [%f,%f]\n',[uu1(imin,jmin); uu2(imin,jmin)])
fprintf('min = %f\n',yy(imin,jmin))
fprintf('\n')

fprintf('Approximated minimum\n');
fprintf('argmin = [%f,%f]\n',xstar)
fprintf('min DLSE = %f\n',net(xstar))
fprintf('min fun = %f\n',dgenfun(xstar(1),xstar(2)))


%%

figure()
subplot(2,1,1)
surf(uu1,uu2,yy)

subplot(2,1,2)
zz = net(u);
surf(uu1,uu2,yy-reshape(zz,size(uu1)))

% matlab2tikz('examp1.tex')