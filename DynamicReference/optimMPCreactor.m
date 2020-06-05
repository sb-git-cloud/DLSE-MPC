%% Benchmark simulation with reactor dynamics given in 

clear all
close all
clc

%% Generate validation data for reactor system

numdatpoints=100;  % # data points
F=1;                % constant input

 % load the networks approximating the MPC
load dlseReactor20000.mat
netPar20 = netPar;
net20 = net;
load dlseReactor25000.mat
netPar25 = netPar;
net25 = net;
load dlseReactor30000.mat
netPar30 = netPar;
net30 = net;

% set optimization options
options = optimoptions('fmincon','OptimalityTolerance', 1e-6,...
    'Display','final','SpecifyObjectiveGradient',true); 

%% Generate state & output trajectory
% Initial conditions
% Cm0=5.3745;
% CI0=.22433;
% D00=3.1308e-3;
% DI0=.62616;
% x0=[Cm0 CI0 D00 DI0]';
% x0 = x0 + x0.*0.5.*randn(4,1);
x0 = [6.6753
    0.3113
    0.0030
    0.6141];

FImin=.003; % min control
FImax=.06;  % max control

width = 30; % fixed pulse width

n = length(x0);

x = zeros(n, numdatpoints+1);
ydr = zeros(1, numdatpoints);
contr = zeros(numdatpoints,1);
compTime = zeros(numdatpoints,1);
Jest = zeros(1, numdatpoints);
x(:,1) = x0;
xt = x0;
N = 5;

tott = [];
totx = [];
yr = [];

rho = 0.9;
epsilon = 0;
kk = 0:N-1;
scaling = (1-epsilon.*(1-rho.^(kk/2))./(1-rho^(1/2)))';

for ii = 1:numdatpoints
    Aeq = eye(n, n + N);
    Beq = xt;
    llb = -(FImax - FImin)/2*scaling;
    uub = (FImax - FImin)/2*scaling;
    mmb = (FImax + FImin)/2;
    uu0 = [xt; mmb + llb + (uub - llb).*rand(N,1)];
    if ii <= floor(numdatpoints/5)
        ydr(ii) = 2e4;
        CnetPar = netPar20;
        Cnet = net20;
    elseif ii <= floor(numdatpoints/3)
        ydr(ii) = 2.5e4;
        CnetPar = netPar25;
        Cnet = net25;
    elseif ii <= floor(numdatpoints/2)
        ydr(ii) = 3e4;
        CnetPar = netPar30;
        Cnet = net30;
    elseif ii <= 2*floor(numdatpoints/3)
        ydr(ii) = 2.5e4;
        CnetPar = netPar25;
        Cnet = net25;
    elseif ii <= 4*floor(numdatpoints/5)
        ydr(ii) = 2e4;
        CnetPar = netPar20;
        Cnet = net20;
    else
        ydr(ii) = 3e4;
        CnetPar = netPar30;
        Cnet = net30;
    end
    tic;
    [xstar,ystar] = dlsea(CnetPar,uu0,[],[],Aeq,Beq,[xt; llb],...
    	[xt; uub],[],options);
    compTime(ii) = toc;
    Jest(ii) = Cnet(xstar);
    contr(ii) = xstar(n+1);
    disp(contr(ii));
    [tt,xx] = ode45(@(t, x) dxdt_reactor(t,x,contr(ii), F), [0 width], xt);
    xt = xx(end,:)';
    x(:,ii+1) = xt;
    cyr = ydr(ii)*ones(length(tt),1);
    tott = [tott; (ii-1)*width + tt];
    totx = [totx; xx];
    yr = [yr;cyr];
end

y = x(4,:)./x(3,:);

%%
N = 5;
R=.2;
Q=1;
trainInp = zeros(n + N, numdatpoints - N + 1);
Jdata = zeros(1, numdatpoints - N + 1);

for kk = 1:numdatpoints - N + 1
    J = 0;
    for kappa = 1:N
        J = J + Q*(y(kk + kappa - 1) - yr(kk))^2 ...
            + R*(contr(kk + kappa - 1))^2;
    end
    Jdata(kk) = J;
end

conty = totx(:,4)./totx(:,3);

%%

figure()
plot(tott, totx)

figure()
stairs(width*(0:numdatpoints-1), contr)

figure()
plot(tott, conty)
hold on
plot(tott, yr,'r')

figure()
plot(width*(0:numdatpoints - N), Jdata)
hold on
plot(width*(0:numdatpoints - N), Jest(1:numdatpoints - N + 1),'r')

%%
fig = figure('Position',[0,0,800,400]);

s1 = subplot(3,1,1);
plot(tott, totx, 'linewidth', 2)
ylabel('state','interpreter','latex','fontsize',18)
legend({'$C_m$','$C_I$','$D_0$','$D_I$'},'interpreter','latex',...
    'location','EastOutside','fontsize',18)
set(gca,'XTickLabel',{})
set(gca,'YTick',[1e-3 1e-1 1e1])
set(gca,'YScale','log')
ylim([1e-4, 5e3])
grid on
box on

s2 = subplot(3,1,2);
stairs(width*(0:numdatpoints), [contr; contr(end)], 'linewidth', 2)
ylabel('control input','interpreter','latex','fontsize',18)
set(gca,'XTickLabel',{})
ylim([FImin, FImax])
grid on
box on

s3 = subplot(3,1,3);
plot(tott, yr,'r--','linewidth',2)
hold on
plot(tott, conty,'linewidth',2)
ylabel('output','interpreter','latex','fontsize',18)
xlabel('time','interpreter','latex','fontsize',18)
legend({'$y_r$','$y$'},'interpreter','latex',...
    'location','EastOutside','fontsize',18)
ylim([0 3.5e4])
ax = gca;
ax.YRuler.Exponent = 0;
grid on
box on

p1 = get(s1,'Position');
p2 = get(s2,'Position');
p3 = get(s3,'Position');

top = p1(2) + p1(4);
bot = p3(2);
hei = (top-bot)/3;

p3n = p3;
p3n(4) = hei;
p2n = p2;
p2n(2) = hei+bot;
p2n(4) = hei;
p1n = p1;
p1n(2) = 2*hei+bot;
p1n(4) = hei;

set(s1,'Position',p1n);
set(s2,'Position',p2n);
set(s3,'Position',p3n);

saveas(fig,'simresults.eps','epsc')

%%

fig2 = figure('Position',[0,0,800,200]);
histogram(compTime,'Normalization','probability')
xlabel('Computing time (s)','interpreter','latex')
ylabel('Time required to perform the DLSEA','interpreter','latex')
saveas(fig2,'compTime.eps','epsc')