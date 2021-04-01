function [dxdt] = dxdt_reactor(~, x, u, F)
% Model after
% Ngoc Thanh Nguyen and Edward Szczerbicki (Eds.) 
%   -Intelligent Systems for Knowledge Management
%    Ch. Neural Networks in MPC page 45/46

% Parameters ------ (see Table 1)
CIin=8;
Cmin=6;
ETc=2.9442e3;
ETd=ETc;
Efm=74478;
EI=1.255e5;
EP=18283;
fstar=.58;
Mm=100.12;
R=8.314;
T=335;
ZTc=3.8223e10;
ZTd=3.1457e11;
Zfm=1.0067e15;
ZI=3.7920e18;
ZP=1.77e9;
V=.1;
% -----------------

% Interpolate control input
FI=u;

% Unwrap state vector
Cm=x(1);
CI=x(2);
D0=x(3);
DI=x(4);

% Side computation
RT=R*T;
P0=sqrt(2*fstar*CI*ZI*exp(-EI/RT)/(ZTd*exp(-ETd/RT)+ZTc*exp(-ETc/RT)));


% Dynamics (eq. (41)-(45))
dCm=-(ZP*exp(-EP/RT)+Zfm*exp(-Efm/RT))*Cm*P0-(F*Cm/V)+(F*Cmin)/V;
dCI=-ZI*exp(-EI/RT)*CI-F*CI/V+FI*CIin/V;
dD0=(.5*ZTc*exp(-ETc/RT)+ZTd*exp(-ETd/RT))*P0^2+Zfm*exp(-Efm/RT)*Cm*P0-F*D0/V;
dDI=Mm*(ZP*exp(-EP/RT)+Zfm*exp(-Efm/RT))*Cm*P0-F*DI/V;

% Output
dxdt=[dCm dCI dD0 dDI]';