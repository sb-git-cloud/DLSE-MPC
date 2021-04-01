function da = forwardprop(dn,n,a,param)
%EXPFUN.FORWARDPROP Forward propagate derivatives from input to output.


  da = bsxfun(@times,dn,a);
end
