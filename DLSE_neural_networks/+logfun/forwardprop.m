function da = forwardprop(dn,n,a,param)
%LOGFUN.FORWARDPROP Forward propagate derivatives from input to output.


  da = bsxfun(@times,dn,1./n);
end
