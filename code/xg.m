function  or3=xg(x, dmodel)

or1 = NaN; or3=[];  or2 = NaN;  dmse = NaN;  % Default return values
[m n] = size(dmodel.S);  % number of design sites and number of dimensions
sx = size(x);            % number of trial sites and their dimension

if  min(sx) == 1 & n > 1 % Single trial point
    nx = max(sx);
    if  nx == n
        mx = 1;  x = x(:).';
    end
else
    mx = sx(1);  nx = sx(2);
end
if  nx ~= n
    error(sprintf('Dimension of trial sites should be %d',n))
end

% Normalize trial sites
x = (x - repmat(dmodel.Ssc(1,:),mx,1)) ./ repmat(dmodel.Ssc(2,:),mx,1);
q = size(dmodel.Ysc,2);  % number of response functions
% Get distances to design sites
dx = zeros(mx*m,n);  kk = 1:m;
for  k = 1 : mx
    dx(kk,:) = repmat(x(k,:),m,1) - dmodel.S;
    kk = kk + m;
end
%% 自相关R（xi xj）
mx1=20;m1=20;
dx1 = zeros(mx1*m1,n);  kk = 1:m1;
for  k = 1 : mx1
    dx1(kk,:) = repmat(x(k,:),m1,1) -x;
    kk = kk + m1;
end
r1 = feval(dmodel.corr, dmodel.theta, dx1);
r1 = reshape(r1, m1, mx1);
%%
% Get regression function and correlation
f = feval(dmodel.regr, x);
r = feval(dmodel.corr, dmodel.theta, dx);
r = reshape(r, m, mx);
% Scaled predictor
% Predictor
rt = dmodel.C \ r;
u = dmodel.G \ (dmodel.Ft.' * rt - f.');
or1 = repmat(dmodel.sigma2,mx,1) .* repmat((1 + colsum(u.^2) - colsum(rt.^2))',1,q);
%%
for i=1:mx
    rt2 =repmat(rt(:,i),1,mx);
    u2 =repmat(u(:,i),1,mx);
    or3(:,i) = repmat(dmodel.sigma2,mx,1) .* repmat((r1(i,:) + colsum(u2.*u) - colsum(rt2.*rt))',1,q);
end
%%
end % of several sites

% >>>>>>>>>>>>>>>>   Auxiliary function  ====================
