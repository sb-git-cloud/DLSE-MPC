%% Benchmark simulation with reactor dynamics given in 
clear all
close all
clc

%% Generate test data for reactor system

numdatpoints=1e5;  % # data points
F=1;                % constant input
yr = 3e4;

%% Generate random input
% Generate input of pulse width random amplitude and fixed width

FImin=.003; % min control
FImax=.06;  % max control
amp=FImin+(FImax-FImin)*rand(numdatpoints,1); % random amplitude
width = 30; % fixed pulse width

%% Generate state & output trajectory
% Initial conditions
Cm0=5.3745;
CI0=.22433;
D00=3.1308e-3;
DI0=.62616;
x0=[Cm0 CI0 D00 DI0]';

n = length(x0);

x = zeros(n, numdatpoints+1);
x(:,1) = x0;
xt = x0;

for ii = 1:numdatpoints
    contr = amp(ii);
    [tt,xx] = ode45(@(t, x) dxdt_reactor(t,x,contr, F), [0 width], xt);
    xt = xx(end,:)';
    x(:,ii+1) = xt;
end

y = x(4,:)./x(3,:);

%%
N = 5;
R=.2;
Q=1;
trainInp = zeros(n + N, numdatpoints - N + 1);
Jdata = zeros(1, numdatpoints - N + 1);

for kk = 1:numdatpoints - N + 1
    trainInp(:,kk) = [x(:,kk); amp(kk:kk+N-1)];
    J = 0;
    for kappa = 1:N
        J = J + Q*(y(kk + kappa - 1) - yr)^2 + R*(amp(kk + kappa - 1))^2;
    end
    Jdata(kk) = J;
end

%%
save(['dataset',num2str(yr),'.mat'],'trainInp','Jdata','yr');