function varargout = dlsea(NetPar,X0,A,B,Aeq,Beq,LB,UB,NONLCON,OPTIONS)
%DLSEA applies the DLSEA to solve a nonconvex optimization problem
%
% [xstar,ystar] = DLSEA(NetPar,X0) starts at X0 and finds a minimum X to 
%  the function synthesized by the DLSE netowrk with parameters NetPar
%
% [xstar,ystar] = DLSEA(NetPar,X0,A,B) starts at X0 and finds a minimum X 
%  to the function synthesized by the DLSE netowrk with parameters NetPar, 
%  subject to the linear inequalities A*X <= B.
%
% [xstar,ystar] = DLSEA(NetPar,X0,A,B,Aeq,Beq) starts at X0 and finds a 
%  minimum X to the function synthesized by the DLSE netowrk with 
%  parameters NetPar, subject to the linear inequalities A*X <= B and to
%  the linear equalitites Aeq*X = Beq.
%
% [xstar,ystar] = DLSEA(NetPar,X0,A,B,Aeq,Beq,LB,UB) starts at X0 and finds 
%  a minimum X to the function synthesized by the DLSE netowrk with 
%  parameters NetPar, subject to the linear inequalities A*X <= B and to
%  the linear equalitites Aeq*X = Beq, with the constraints LB <= X <= UB.
%
% [xstar,ystar] = DLSEA(NetPar,X0,A,B,Aeq,Beq,LB,UB,NONLCON,OPTIONS) 
%  starts at X0 and finds  a minimum X to the function synthesized by the 
%  DLSE netowrk with parameters NetPar, subject to the linear inequalities 
%  A*X <= B and to the linear equalitites Aeq*X = Beq, with the constraints 
%  LB <= X <= UB. The minimization is subject to the constraints defined 
%  in NONLCON. The default options are replaced by the values in OPTIONS.
%
% [xstar,ystar] = DLSEA(NetPar,X0,A,B,Aeq,Beq,LB,UB,NONLCON) 
%  starts at X0 and finds  a minimum X to the function synthesized by the 
%  DLSE netowrk with parameters NetPar, subject to the linear inequalities 
%  A*X <= B and to the linear equalitites Aeq*X = Beq, with the constraints 
%  LB <= X <= UB. The minimization is subject to the constraints defined 
%  in NONLCON.

switch nargin
    case 0
        error 'the function requires the parameters of the trained network'
    case 1
        opt = optimoptions('fmincon'); 
        Alpha2 = NetPar.Alpha2;
        X0 = zeros(size(Alpha2,2),1);
    case 10
        opt = OPTIONS; 
    otherwise 
        opt = optimoptions('fmincon'); 
end

Alpha1 = NetPar.Alpha1;
Beta1 = NetPar.Beta1;
Alpha2 = NetPar.Alpha2;
Beta2 = NetPar.Beta2;
Dugain = diag(NetPar.ugain);
ygain = NetPar.ygain;
uoff = NetPar.uoff;
umin = NetPar.umin;
ymin = NetPar.ymin;
yoff = NetPar.yoff;

g = @(x) ( log(sum(exp(Alpha1*(Dugain*(x - uoff) + umin) + Beta1),1)) - ymin/2)/ygain + yoff/2;     

dh = @(x) (ygain * sum(exp(Alpha2*(Dugain*(x - uoff) + umin) + Beta2),1))^(-1)*...
    sum(exp(Alpha2*(Dugain*(x - uoff) + umin) + Beta2).*Alpha2*Dugain,1)';

gstar = @(x,yk) g(x) - (x')*yk; 

xk = X0;

while true
    yk = dh(xk);
    switch nargin
        case {1,2}
            xk1 = fmincon(@(x) gstar(x,yk),xk);
        case 4
            xk1 = fmincon(@(x) gstar(x,yk),xk,A,B);
        case 6
            xk1 = fmincon(@(x) gstar(x,yk),xk,A,B,Aeq,Beq);
        case 8
            xk1 = fmincon(@(x) gstar(x,yk),xk,A,B,Aeq,Beq,LB,UB);
        case 9
            xk1 = fmincon(@(x) gstar(x,yk),xk,A,B,Aeq,Beq,LB,UB,NONLCON);
        case 10
            obj = @(x) funwithgrad(x,Alpha1,Beta1,Dugain,uoff,umin,ymin,yoff,ygain,yk);
            xk1 = fmincon(obj,xk,A,B,Aeq,Beq,LB,UB,NONLCON,OPTIONS);
        otherwise
            error 'wrong number of inputs'
    end
    if norm(xk1-xk)/(1+norm(xk)) <= opt.OptimalityTolerance
        break;
    else
        xk = xk1;
    end
end

switch nargout
    case 1 
        varargout{1} = xk;
    case 2
        varargout{1} = xk;
        varargout{2} = yk;
end

end

function [gstar,gradg] = funwithgrad(x,Alpha1,Beta1,Dugain,uoff,umin,ymin,yoff,ygain,yk)

grad = (ygain * sum(exp(Alpha1*(Dugain*(x - uoff) + umin) + Beta1),1))^(-1)*...
           sum(exp(Alpha1*(Dugain*(x - uoff) + umin) + Beta1).*Alpha1*Dugain,1)';
g = (log(sum(exp(Alpha1*(Dugain*(x - uoff) + umin) + Beta1),1)) - ymin/2)/ygain + yoff/2; 
gstar = g - (x')*yk; 

if nargout > 1
    gradg = grad - yk;
end

end