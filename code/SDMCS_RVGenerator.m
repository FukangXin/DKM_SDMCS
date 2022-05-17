function SPool =  SDMCS_RVGenerator(p1,p2,Ndim,NoS)
% function name: SDMCS Random Vector Generator
% function: generate samples obeying PDF fi(x)
%          where fi(x) = normpdf(0,1) ,x ¡Ê Di ; fi(x) = 0,  x ~¡Ê Di
% input:   Cumulative distribution probability: p1 and p2
%          Number of dimension: Ndim
%          Number of Samples: NoS
% output:  Sample Pool£º sample
%% Check input Parameter
if p1<0||p1>1||p2<0||p2>1
    error('please input p1 or p2 between [0,1]')
end

if p1 >= p2
    error('please make sure p2 > p1');
end
%% example
% p1 = 0.9999999
% p2 =1
% Ndim = 2
% NoS = 1000000
%%
    % 1. Generate uniform distribution U(p1,p2)
        p = p1 + (p2-p1) * rand(NoS,1);

    % 2. Tranfer into chi2square distribution
        R = sqrt(chi2inv(p,Ndim));
     
    % 3. Generate unit uniform random vector    
        V = normrnd(0,1,NoS,Ndim);
        V = V./pdist2(V,zeros(1,Ndim));
        
    % 4. Samples obeying target ditribution 
        SPool = R*ones(1,Ndim).*V;
end
