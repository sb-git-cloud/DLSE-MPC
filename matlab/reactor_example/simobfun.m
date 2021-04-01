function [J,y] = simobfun(xt,u,yr,F,N,R,Q,width)

y = zeros(1,N);
y(1) = xt(4,1)/xt(3,1);
J = Q*(y(1) - yr)^2 + R*u(1)^2;

for kk = 2:N
    [~,xx] = ode45(@(t, x) dxdt_reactor(t,x,u(kk-1), F), [0 width], xt);
    xt = xx(end,:)';
    y(kk) = xt(4,1)/xt(3,1);
    J = J + Q*(y(kk) - yr)^2 + R*u(kk)^2;
end
