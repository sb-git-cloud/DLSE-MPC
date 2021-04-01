function dn = backprop(da,n,a,param)
%LOGFUN.BACKPROP Backpropagate derivatives from outputs to inputs

  dn = bsxfun(@times,da,1./n);
end
