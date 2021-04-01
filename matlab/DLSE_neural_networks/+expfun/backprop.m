function dn = backprop(da,n,a,param)
%EXPFUN.BACKPROP Backpropagate derivatives from outputs to inputs


  dn = bsxfun(@times,da,a);
end
