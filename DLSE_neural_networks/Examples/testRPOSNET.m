clear all
close all
clc

m = 100;
noi = 0; %0.1;

u = rand(1,m);

dgenfun = @(u) 2*u.^2-u.^3+3*cos(u);

y = dgenfun(u);
y = y + noi*randn(size(y));

figure()
subplot(3,1,1)
plot(u,y,'r*')

net = feedforwardnet(10,'trainlm');
tpam = net.trainParam;
tpam.max_fail = 1e3;

[net, Temp, netPar, gpos, fun] = trainRPOS(u,y,10,'trainlm',tpam);

uu = linspace(min(u),max(u));
hold on
plot(uu,dgenfun(uu),'k--')
% plot(uu, gpos(uu), 'c', 'linewidth', 1.2)
rfun = @(x) (fun.num(x))./(fun.den(x));
plot(uu, rfun(uu), 'b')

subplot(3,1,2)
plot(uu, fun.num(uu))

subplot(3,1,3)
plot(uu, fun.den(uu))