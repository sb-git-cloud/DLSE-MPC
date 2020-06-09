function a = logfun(n,varargin)
%LOGFUN logarithmic transfer function.
%
% Transfer functions convert a neural network layer's net input into
% its net output.	

if ischar(n)
  a = nnet7.transfer_fcn(mfilename,n,varargin{:});
  return
end

% Apply
a = logfun.apply(n);

