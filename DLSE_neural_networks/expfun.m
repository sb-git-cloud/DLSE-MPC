function a = expfun(n,varargin)
%EXPFUN exponential function.
%
% Transfer functions convert a neural network layer's net input into
% its net output.	

if ischar(n)
  a = nnet7.transfer_fcn(mfilename,n,varargin{:});
  return
end

% Apply
a = expfun.apply(n);

